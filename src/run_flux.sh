#!/bin/bash

# =============================================================================
# FLUX Artifact Generation Pipeline Script
# =============================================================================
# This script runs FLUX artifact generation using pre-processed GSAM results
# to create visual artifacts for training data synthesis.
#
# Usage: ./run_flux.sh [gsam_output_dir] [options]
# Example: ./run_flux.sh gsam_output_person --artifact-types "distortion removal"
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
GSAM_DIR=""
ARTIFACT_TYPES="addition removal distortion"
DEVICE="cuda"
RESUME=false
OUTPUT_DIR=""
INJECT=15
USE_RF_SOLVER=false
PE_STEP_ADDITION=0.3
PE_STEP_REMOVAL=0.5
PE_STEP_DISTORTION=0.7
GUIDANCE=5.0
NUM_STEPS=25

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
FLUX Artifact Generation Pipeline

USAGE:
    ./run_flux.sh <gsam_output_dir> [OPTIONS]

ARGUMENTS:
    gsam_output_dir         Directory containing GSAM segmentation results
                           (must contain processed_data/ subdirectory)

OPTIONS:
    --artifact-types LIST   Artifact types to generate (default: "distortion removal addition")
                           Available types: distortion, removal, addition
    --device DEVICE         Device to use (default: cuda)
    --output-dir DIR        Output directory (default: flux_output_<supercategory>)
    --resume                Resume processing from previous run
    --inject INT            Inject step for FLUX generation (default: 15)
    --pe-step-addition FLOAT PE step value for addition artifacts (default: 0.3)
    --pe-step-removal FLOAT  PE step value for removal artifacts (default: 0.5)
    --pe-step-distortion FLOAT PE step value for distortion artifacts (default: 0.7)
    --guidance FLOAT        FLUX guidance value (default: 5.0)
    --num-steps INT         Number of FLUX generation steps (default: 25)
    --use-rf-solver         Use RF solver (second-order) instead of first-order denoising (default: False)
    --help                  Show this help message

EXAMPLES:
    # Basic usage
    ./run_flux.sh gsam_output_person

    # Generate only distortion artifacts
    ./run_flux.sh gsam_output_person --artifact-types "distortion"

    # Use CPU and custom output directory
    ./run_flux.sh gsam_output_animal --device cpu --output-dir custom_flux_results

    # Resume interrupted processing
    ./run_flux.sh gsam_output_person --resume

    # Custom inject step value
    ./run_flux.sh gsam_output_person --inject 30

    # Custom PE step values for fine-tuning
    ./run_flux.sh gsam_output_person --pe-step-addition 0.2 --pe-step-removal 0.6 --pe-step-distortion 0.8

    # Adjust FLUX parameters
    ./run_flux.sh gsam_output_person --guidance 7.5 --num-steps 30

    # Use RF solver for more accurate generation
    ./run_flux.sh gsam_output_person --use-rf-solver

REQUIREMENTS:
    - GSAM results directory with processed_data/ subdirectory
    - OPENAI_API_KEY environment variable set
    - Python environment with required FLUX dependencies

OUTPUT STRUCTURE:
    flux_output_<supercategory>/       # FLUX generation results
    ├── <image_name>/                  # Results for each image
    │   ├── 01_original_image.png
    │   ├── 02_detection_results.png
    │   ├── 03_bbox_overlay_*.png
    │   ├── 04_comparison_*.png
    │   ├── 05_artifact_comparison_*.png
    │   └── 06_masks_*.png
    └── logs/                          # Processing logs

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --artifact-types)
            ARTIFACT_TYPES="$2"
            shift 2
            ;;
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
        --pe-step-addition)
            PE_STEP_ADDITION="$2"
            shift 2
            ;;
        --pe-step-removal)
            PE_STEP_REMOVAL="$2"
            shift 2
            ;;
        --pe-step-distortion)
            PE_STEP_DISTORTION="$2"
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
            if [[ -z "$GSAM_DIR" ]]; then
                GSAM_DIR="$1"
            else
                print_error "Multiple GSAM directories specified: $GSAM_DIR and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$GSAM_DIR" ]]; then
    print_error "GSAM output directory is required"
    show_help
    exit 1
fi

# Verify GSAM directory exists and has required structure
if [[ ! -d "$GSAM_DIR" ]]; then
    print_error "GSAM directory does not exist: $GSAM_DIR"
    exit 1
fi

if [[ ! -d "$GSAM_DIR/processed_data" ]]; then
    print_error "GSAM intermediate data directory not found: $GSAM_DIR/processed_data"
    print_error "Make sure to run GSAM segmentation first with run_gsam.sh"
    exit 1
fi

# Set output directory if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    # Extract supercategory from GSAM directory name
    basename_gsam=$(basename "$GSAM_DIR")
    if [[ "$basename_gsam" =~ ^gsam_output_(.+)$ ]]; then
        supercategory="${BASH_REMATCH[1]}"
        OUTPUT_DIR="flux_output_${supercategory}"
    else
        OUTPUT_DIR="flux_output_results"
    fi
fi

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check required Python script
    if [[ ! -f "batch_flux_generation.py" ]]; then
        print_error "batch_flux_generation.py not found in current directory"
        exit 1
    fi
    
    # Check OpenAI API key
    if [[ -z "$OPENAI_API_KEY" ]]; then
        print_error "OPENAI_API_KEY environment variable is not set"
        print_info "Please set it with: export OPENAI_API_KEY='your-api-key'"
        exit 1
    fi
    
    # Check GSAM intermediate data
    intermediate_count=$(find "$GSAM_DIR/processed_data" -name "image_*.pkl" 2>/dev/null | wc -l)
    if [[ $intermediate_count -eq 0 ]]; then
        print_error "No GSAM intermediate data files found in $GSAM_DIR/processed_data"
        print_error "Make sure GSAM processing completed successfully"
        exit 1
    fi
    
    print_success "Dependencies check passed"
    print_info "Found $intermediate_count GSAM intermediate files to process"
}

# Print configuration
print_configuration() {
    print_header "FLUX ARTIFACT GENERATION CONFIGURATION"
    echo "GSAM directory:        $GSAM_DIR"
    echo "Artifact types:        $ARTIFACT_TYPES"
    echo "Device:                $DEVICE"
    echo "Output directory:      $OUTPUT_DIR"
    echo "Inject step:           $INJECT"
    echo "PE step addition:      $PE_STEP_ADDITION"
    echo "PE step removal:       $PE_STEP_REMOVAL"
    echo "PE step distortion:    $PE_STEP_DISTORTION"
    echo "Guidance:              $GUIDANCE"
    echo "Number of steps:       $NUM_STEPS"
    echo "Use RF solver:         $USE_RF_SOLVER"
    echo ""
}

# Run FLUX artifact generation
run_flux_generation() {
    print_header "FLUX ARTIFACT GENERATION PROCESSING"
    
    intermediate_count=$(find "$GSAM_DIR/processed_data" -name "image_*.pkl" | wc -l)
    print_info "Processing $intermediate_count intermediate data files with FLUX"
    
    # Build FLUX command
    flux_cmd="python batch_flux_generation.py $GSAM_DIR"
    flux_cmd="$flux_cmd --artifact-types $ARTIFACT_TYPES"
    flux_cmd="$flux_cmd --device $DEVICE"
    flux_cmd="$flux_cmd --output-dir $OUTPUT_DIR"
    flux_cmd="$flux_cmd --inject $INJECT"
    flux_cmd="$flux_cmd --pe-step-addition $PE_STEP_ADDITION"
    flux_cmd="$flux_cmd --pe-step-removal $PE_STEP_REMOVAL"
    flux_cmd="$flux_cmd --pe-step-distortion $PE_STEP_DISTORTION"
    flux_cmd="$flux_cmd --guidance $GUIDANCE"
    flux_cmd="$flux_cmd --num-steps $NUM_STEPS"
    
    if [[ "$RESUME" == true ]]; then
        flux_cmd="$flux_cmd --resume"
    fi
    
    if [[ "$USE_RF_SOLVER" == true ]]; then
        flux_cmd="$flux_cmd --use-rf-solver"
    fi
    
    print_info "Running FLUX artifact generation..."
    print_info "Command: $flux_cmd"
    echo ""
    
    # Run FLUX generation
    start_time=$(date +%s)
    if $flux_cmd; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        print_success "FLUX artifact generation completed in ${elapsed}s"
    else
        print_error "FLUX artifact generation failed"
        exit 1
    fi
    
    # Verify FLUX output
    if [[ -d "$OUTPUT_DIR" ]]; then
        result_count=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
        print_info "Generated $result_count visualization files"
    else
        print_warning "FLUX output directory not found: $OUTPUT_DIR"
    fi
}

# Print final summary
print_summary() {
    print_header "FLUX ARTIFACT GENERATION SUMMARY"
    
    # Input summary
    intermediate_count=$(find "$GSAM_DIR/processed_data" -name "image_*.pkl" 2>/dev/null | wc -l)
    echo "Input (GSAM Results):"
    echo "  Source directory:     $GSAM_DIR"
    echo "  Intermediate files:   $intermediate_count"
    echo ""
    
    # FLUX results
    if [[ -d "$OUTPUT_DIR" ]]; then
        visualization_count=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
        image_dirs=$(find "$OUTPUT_DIR" -maxdepth 1 -type d ! -name "logs" ! -path "$OUTPUT_DIR" | wc -l)
        echo "Output (FLUX Results):"
        echo "  Output directory:     $OUTPUT_DIR"
        echo "  Processed images:     $image_dirs"
        echo "  Visualization files:  $visualization_count"
        echo ""
    fi
    
    # Log files
    echo "Log Files:"
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        echo "  FLUX logs:            $OUTPUT_DIR/logs/"
    fi
    echo ""
    
    print_success "FLUX artifact generation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  - Review the generated visualizations and artifacts"
    echo "  - Check the log files for detailed processing information"
    echo "  - Use the generated artifacts for your downstream ML tasks"
    echo "  - Analyze the artifact quality and adjust parameters if needed"
}

# Cleanup function for interrupted execution
cleanup() {
    print_warning "FLUX processing interrupted"
    print_info "Partial results may be available in output directory: $OUTPUT_DIR"
    print_info "Use --resume to continue from where you left off"
    exit 1
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_header "FLUX ARTIFACT GENERATION PIPELINE"
    echo "Starting FLUX processing from GSAM results: $GSAM_DIR"
    echo "Timestamp: $(date)"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Print configuration
    print_configuration
    
    # Record start time
    start_time=$(date +%s)
    
    # Run FLUX generation
    run_flux_generation
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