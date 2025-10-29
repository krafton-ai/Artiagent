#!/bin/bash

# =============================================================================
# FLUX Image Reconstruction Pipeline Script
# =============================================================================
# This script runs FLUX image reconstruction on directories containing 
# real_image.png files without injecting any artifacts.
#
# Usage: ./run_flux_recon.sh [input_dir] [options]
# Example: ./run_flux_recon.sh my_images_dir --inject 20
# =============================================================================

set -e  # Exit on any error
if [[ -f .env ]]; then
    source .env
fi
if [[ -f ../.env ]]; then
    source ../.env
fi

# Ensure OPENAI_API_KEY is exported to subprocesses
if [[ -n "$OPENAI_API_KEY" ]]; then
    export OPENAI_API_KEY
fi

# Default values
INPUT_DIR=""
DEVICE="cuda"
RESUME=false
OUTPUT_DIR=""
INJECT=25
USE_RF_SOLVER=false
GUIDANCE=5.0
NUM_STEPS=25
MULTI_GPU=false
NUM_GPUS=""  # Auto-detect if not specified

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo -e "${NC}"
}

# Function to show usage
show_help() {
    cat << EOF
FLUX Image Reconstruction Pipeline

USAGE:
    ./run_flux_recon.sh <input_dir> [OPTIONS]

ARGUMENTS:
    input_dir               Directory containing subdirectories with real_image.png files
                           (each subdirectory should contain one real_image.png)

OPTIONS:
    --device DEVICE         Device to use (default: cuda)
    --output-dir DIR        Output directory (default: flux_reconstruction_<dirname>)
    --resume                Resume processing from previous run
    --inject INT            Inject step for FLUX generation (default: 25)
    --guidance FLOAT        FLUX guidance value (default: 5.0)
    --num-steps INT         Number of FLUX generation steps (default: 25)
    --use-rf-solver         Use RF solver (second-order) instead of first-order denoising (default: False)
    --multi-gpu             Enable multi-GPU parallelization (auto-detects available GPUs)
    --num-gpus INT          Number of GPUs to use (default: auto-detect all available)
    --help                  Show this help message

EXAMPLES:
    # Basic usage
    ./run_flux_recon.sh my_images

    # Use CPU and custom output directory
    ./run_flux_recon.sh my_images --device cpu --output-dir custom_results

    # Resume interrupted processing
    ./run_flux_recon.sh my_images --resume

    # Custom inject step value
    ./run_flux_recon.sh my_images --inject 20

    # Adjust FLUX parameters
    ./run_flux_recon.sh my_images --guidance 7.5 --num-steps 30

    # Use RF solver for more accurate generation
    ./run_flux_recon.sh my_images --use-rf-solver

    # Enable multi-GPU parallelization (auto-detect all GPUs)
    ./run_flux_recon.sh my_images --multi-gpu

    # Use specific number of GPUs
    ./run_flux_recon.sh my_images --multi-gpu --num-gpus 4

REQUIREMENTS:
    - Input directory with subdirectories containing real_image.png files
    - Python environment with required FLUX dependencies

OUTPUT STRUCTURE:
    flux_reconstruction_<dirname>/     # FLUX reconstruction results
    ├── <subdir_name>/                 # Results for each image
    │   ├── real_image.png
    │   ├── reconstructed_image.png
    │   └── comparison.png
    └── logs/                          # Processing logs

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --inject)
            INJECT="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --use-rf-solver)
            USE_RF_SOLVER=true
            shift
            ;;
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            else
                print_error "Multiple input directories specified: $INPUT_DIR and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]]; then
    print_error "Input directory is required"
    show_help
    exit 1
fi

# Verify input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    print_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Print INPUT_DIR for debugging
print_info "Input directory: $INPUT_DIR"

# Check for subdirectories with real_image.png files
print_info "Checking for subdirectories with real_image.png files..."

# Count real_image.png files in subdirectories
image_count=$(find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -name "real_image.png" -type f | wc -l)
print_info "Found $image_count real_image.png files"

if [[ $image_count -eq 0 ]]; then
    print_error "No real_image.png files found in subdirectories of: $INPUT_DIR"
    print_error "Expected structure: $INPUT_DIR/<subdir>/real_image.png"
    
    # Show what directories exist for debugging
    print_info "Available subdirectories in $INPUT_DIR:"
    ls -la "$INPUT_DIR" | grep "^d" || print_info "No subdirectories found"
    exit 1
fi

print_info "Validation passed! Found $image_count images. Proceeding with FLUX reconstruction..."

# Set output directory if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    # Extract directory name for output naming
    basename_input=$(basename "$INPUT_DIR")
    OUTPUT_DIR="flux_reconstruction_${basename_input}"
fi

# Detect available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        # Count number of GPUs
        local gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        echo "$gpu_count"
    else
        echo "0"
    fi
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check required Python script
    if [[ ! -f "batch_flux_reconstruction.py" ]]; then
        print_error "batch_flux_reconstruction.py not found in current directory"
        exit 1
    fi
    
    # Check input data (subdirectories with real_image.png files)
    image_count=$(find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -name "real_image.png" -type f 2>/dev/null | wc -l)
    if [[ $image_count -eq 0 ]]; then
        print_error "No real_image.png files found in $INPUT_DIR subdirectories"
        print_error "Expected structure: $INPUT_DIR/<subdir>/real_image.png"
        exit 1
    fi
    
    # Check GPU availability if multi-GPU mode is enabled
    if [[ "$MULTI_GPU" == true ]]; then
        local available_gpus=$(detect_gpus)
        if [[ $available_gpus -eq 0 ]]; then
            print_error "Multi-GPU mode enabled but no GPUs detected"
            print_error "Make sure nvidia-smi is available and GPUs are visible"
            exit 1
        fi
        print_info "Detected $available_gpus GPU(s) available"
        
        # Set NUM_GPUS if not specified
        if [[ -z "$NUM_GPUS" ]]; then
            NUM_GPUS=$available_gpus
            print_info "Using all $NUM_GPUS GPU(s)"
        elif [[ $NUM_GPUS -gt $available_gpus ]]; then
            print_warning "Requested $NUM_GPUS GPUs but only $available_gpus available"
            NUM_GPUS=$available_gpus
            print_info "Using $NUM_GPUS GPU(s)"
        fi
    fi
    
    print_success "Dependencies check passed"
    print_info "Found $image_count images to process"
}

# Print configuration
print_configuration() {
    print_header "FLUX IMAGE RECONSTRUCTION CONFIGURATION"
    echo "Input directory:       $INPUT_DIR"
    echo "Device:                $DEVICE"
    echo "Output directory:      $OUTPUT_DIR"
    echo "Inject step:           $INJECT"
    echo "Guidance:              $GUIDANCE"
    echo "Number of steps:       $NUM_STEPS"
    echo "Use RF solver:         $USE_RF_SOLVER"
    if [[ "$MULTI_GPU" == true ]]; then
        echo "Multi-GPU mode:        ENABLED"
        echo "Number of GPUs:        $NUM_GPUS"
    else
        echo "Multi-GPU mode:        DISABLED"
    fi
    echo ""
}

# Run FLUX image reconstruction
run_flux_reconstruction() {
    print_header "FLUX IMAGE RECONSTRUCTION PROCESSING"
    
    image_count=$(find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -name "real_image.png" -type f | wc -l)
    print_info "Processing $image_count images with FLUX reconstruction"
    
    start_time=$(date +%s)
    
    if [[ "$MULTI_GPU" == true ]]; then
        # Multi-GPU parallel processing
        print_info "Launching $NUM_GPUS parallel processes (one per GPU)..."
        
        # Array to store background process PIDs
        declare -a pids
        
        # Launch one process per GPU
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            # Build FLUX command for this GPU
            flux_cmd="python batch_flux_reconstruction.py $INPUT_DIR"
            flux_cmd="$flux_cmd --device $DEVICE"
            flux_cmd="$flux_cmd --output-dir $OUTPUT_DIR"
            flux_cmd="$flux_cmd --inject $INJECT"
            flux_cmd="$flux_cmd --guidance $GUIDANCE"
            flux_cmd="$flux_cmd --num-steps $NUM_STEPS"
            flux_cmd="$flux_cmd --gpu-id $gpu_id"
            flux_cmd="$flux_cmd --total-gpus $NUM_GPUS"
            
            if [[ "$RESUME" == true ]]; then
                flux_cmd="$flux_cmd --resume"
            fi
            
            if [[ "$USE_RF_SOLVER" == true ]]; then
                flux_cmd="$flux_cmd --use-rf-solver"
            fi
            
            print_info "GPU $gpu_id: Starting process..."
            print_info "  Command: $flux_cmd"
            
            # Run in background and capture PID
            $flux_cmd > "$OUTPUT_DIR/logs/gpu_${gpu_id}.log" 2>&1 &
            pids[$gpu_id]=$!
            
            # Small delay to avoid race conditions
            sleep 2
        done
        
        echo ""
        print_info "All $NUM_GPUS GPU processes launched"
        print_info "Waiting for completion..."
        echo ""
        
        # Wait for all processes and check exit status
        failed_gpus=0
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            pid=${pids[$gpu_id]}
            print_info "Waiting for GPU $gpu_id process (PID: $pid)..."
            
            if wait $pid; then
                print_success "GPU $gpu_id: Completed successfully"
            else
                print_error "GPU $gpu_id: Failed (PID: $pid)"
                ((failed_gpus++))
            fi
        done
        
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        
        echo ""
        if [[ $failed_gpus -eq 0 ]]; then
            print_success "All GPU processes completed successfully in ${elapsed}s"
        else
            print_error "$failed_gpus GPU process(es) failed"
            print_info "Check individual GPU logs in $OUTPUT_DIR/logs/"
            exit 1
        fi
        
    else
        # Single GPU processing (original behavior)
        # Build FLUX command
        flux_cmd="python batch_flux_reconstruction.py $INPUT_DIR"
        flux_cmd="$flux_cmd --device $DEVICE"
        flux_cmd="$flux_cmd --output-dir $OUTPUT_DIR"
        flux_cmd="$flux_cmd --inject $INJECT"
        flux_cmd="$flux_cmd --guidance $GUIDANCE"
        flux_cmd="$flux_cmd --num-steps $NUM_STEPS"
        
        if [[ "$RESUME" == true ]]; then
            flux_cmd="$flux_cmd --resume"
        fi
        
        if [[ "$USE_RF_SOLVER" == true ]]; then
            flux_cmd="$flux_cmd --use-rf-solver"
        fi
        
        print_info "Running FLUX image reconstruction..."
        print_info "Command: $flux_cmd"
        echo ""
        
        # Run FLUX reconstruction
        if $flux_cmd; then
            end_time=$(date +%s)
            elapsed=$((end_time - start_time))
            print_success "FLUX image reconstruction completed in ${elapsed}s"
        else
            print_error "FLUX image reconstruction failed"
            exit 1
        fi
    fi
    
    # Verify FLUX output
    if [[ -d "$OUTPUT_DIR" ]]; then
        result_count=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
        print_info "Generated $result_count output files"
    else
        print_warning "FLUX output directory not found: $OUTPUT_DIR"
    fi
}

# Merge progress files from multi-GPU runs
merge_progress_files() {
    if [[ "$MULTI_GPU" != true ]]; then
        return
    fi
    
    print_info "Merging progress files from $NUM_GPUS GPUs..."
    
    # Python script to merge progress files
    python3 - "$OUTPUT_DIR" "$NUM_GPUS" << 'PYTHON_SCRIPT'
import json
import sys
import os
from pathlib import Path

output_dir = sys.argv[1]
num_gpus = int(sys.argv[2])

merged_stats = {
    'total_images': 0,
    'processed_images': 0,
    'successful_images': 0,
    'failed_images': 0,
    'processed_image_ids': [],
    'start_time': None,
    'gpu_stats': []
}

for gpu_id in range(num_gpus):
    progress_file = Path(output_dir) / f'flux_reconstruction_progress_gpu{gpu_id}.json'
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                gpu_stats = json.load(f)
                merged_stats['total_images'] += gpu_stats.get('total_images', 0)
                merged_stats['processed_images'] += gpu_stats.get('processed_images', 0)
                merged_stats['successful_images'] += gpu_stats.get('successful_images', 0)
                merged_stats['failed_images'] += gpu_stats.get('failed_images', 0)
                merged_stats['processed_image_ids'].extend(gpu_stats.get('processed_image_ids', []))
                
                # Track per-GPU stats
                merged_stats['gpu_stats'].append({
                    'gpu_id': gpu_id,
                    'processed': gpu_stats.get('processed_images', 0),
                    'successful': gpu_stats.get('successful_images', 0),
                    'failed': gpu_stats.get('failed_images', 0)
                })
                
                # Use earliest start time
                if merged_stats['start_time'] is None or gpu_stats.get('start_time', '') < merged_stats['start_time']:
                    merged_stats['start_time'] = gpu_stats.get('start_time')
        except Exception as e:
            print(f"Warning: Could not read progress file for GPU {gpu_id}: {e}", file=sys.stderr)

# Save merged progress
output_file = Path(output_dir) / 'flux_reconstruction_progress_merged.json'
with open(output_file, 'w') as f:
    json.dump(merged_stats, f, indent=2)

print(f"Merged progress saved to: {output_file}")
PYTHON_SCRIPT
    
    print_success "Progress files merged"
}

# Print final summary
print_summary() {
    print_header "FLUX IMAGE RECONSTRUCTION SUMMARY"
    
    # Merge progress files if multi-GPU
    if [[ "$MULTI_GPU" == true ]]; then
        merge_progress_files
    fi
    
    # Input summary
    image_count=$(find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -name "real_image.png" -type f 2>/dev/null | wc -l)
    echo "Input:"
    echo "  Source directory:     $INPUT_DIR"
    echo "  Input images:         $image_count"
    echo ""
    
    # FLUX results
    if [[ -d "$OUTPUT_DIR" ]]; then
        output_count=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
        image_dirs=$(find "$OUTPUT_DIR" -maxdepth 1 -type d ! -name "logs" ! -path "$OUTPUT_DIR" | wc -l)
        echo "Output (FLUX Reconstruction):"
        echo "  Output directory:     $OUTPUT_DIR"
        echo "  Processed images:     $image_dirs"
        echo "  Output files:         $output_count"
        
        # Show per-GPU stats if available
        if [[ "$MULTI_GPU" == true ]] && [[ -f "$OUTPUT_DIR/flux_reconstruction_progress_merged.json" ]]; then
            echo ""
            echo "  Per-GPU Statistics:"
            for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
                if [[ -f "$OUTPUT_DIR/flux_reconstruction_progress_gpu${gpu_id}.json" ]]; then
                    stats=$(python3 -c "import json; f=open('$OUTPUT_DIR/flux_reconstruction_progress_gpu${gpu_id}.json'); d=json.load(f); print(f\"{d.get('successful_images',0)}/{d.get('processed_images',0)}\")")
                    echo "    GPU $gpu_id:            $stats successful"
                fi
            done
        fi
        echo ""
    fi
    
    # Log files
    echo "Log Files:"
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        echo "  FLUX logs:            $OUTPUT_DIR/logs/"
        if [[ "$MULTI_GPU" == true ]]; then
            echo "  Per-GPU logs:         $OUTPUT_DIR/logs/gpu_*.log"
        fi
    fi
    echo ""
    
    print_success "FLUX image reconstruction completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  - Review the reconstructed images and comparisons"
    echo "  - Check the log files for detailed processing information"
    echo "  - Analyze the reconstruction quality"
    echo "  - Adjust parameters if needed for better results"
}

# Cleanup function for interrupted execution
cleanup() {
    print_warning "FLUX processing interrupted"
    
    # Kill all background GPU processes if in multi-GPU mode
    if [[ "$MULTI_GPU" == true ]] && [[ -n "${pids[@]}" ]]; then
        print_info "Stopping all GPU processes..."
        for pid in "${pids[@]}"; do
            if kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null
            fi
        done
    fi
    
    print_info "Partial results may be available in output directory: $OUTPUT_DIR"
    print_info "Use --resume to continue from where you left off"
    exit 1
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_header "FLUX IMAGE RECONSTRUCTION PIPELINE"
    echo "Starting FLUX reconstruction from: $INPUT_DIR"
    echo "Timestamp: $(date)"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Print configuration
    print_configuration
    
    # Record start time
    start_time=$(date +%s)
    
    # Run FLUX reconstruction
    run_flux_reconstruction
    echo ""
    
    # Calculate total time
    end_time=$(date +%s)
    total_elapsed=$((end_time - start_time))
    
    # Print summary
    print_summary
    print_info "Total processing time: ${total_elapsed}s ($(date -u -d @${total_elapsed} +%H:%M:%S))"
}

# Run main function
main "$@" 