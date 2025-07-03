#!/bin/bash

# =============================================================================
# VLPart Segmentation Pipeline Script
# =============================================================================
# This script runs VLPart segmentation batch processing to generate part 
# segmentation masks and annotations.
#
# Usage: ./run_vlpart.sh [supercategory] [options]
# Example: ./run_vlpart.sh person --max-images 100 --device cuda
# =============================================================================

set -e  # Exit on any error

# Default values
SUPERCATEGORY=""
ARTIFACT_TYPES="distortion removal addition"
MAX_IMAGES=""
MIN_AREA_RATIO="0.05"
MAX_AREA_RATIO="0.5"
DEVICE="cuda"
RESUME=false
OUTPUT_DIR=""
PREDEFINED_VOCAB=""

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
VLPart Segmentation Pipeline

USAGE:
    ./run_vlpart.sh <supercategory> [OPTIONS]

ARGUMENTS:
    supercategory           COCO supercategory to process (person, animal, vehicle, etc.)

OPTIONS:
    --artifact-types LIST   Artifact types to generate (default: "distortion removal addition")
    --max-images N          Maximum number of images to process
    --min-area-ratio FLOAT  Minimum area ratio for part filtering (default: 0.2)
    --max-area-ratio FLOAT  Maximum area ratio for part filtering (default: 0.5)
    --device DEVICE         Device to use (default: cuda)
    --output-dir DIR        Output directory (default: vlpart_output_<supercategory>)
    --resume                Resume processing from previous run
    --predefined-vocab LIST Pre-defined vocabulary (e.g., --predefined-vocab "person head" "person arm")
    --help                  Show this help message

EXAMPLES:
    # Basic usage
    ./run_vlpart.sh person

    # Process 100 images with custom filtering
    ./run_vlpart.sh person --max-images 100 --min-area-ratio 0.02 --max-area-ratio 0.6

    # Only generate distortion artifacts
    ./run_vlpart.sh animal --artifact-types "distortion"

    # Resume interrupted processing
    ./run_vlpart.sh person --resume

    # Custom output directory
    ./run_vlpart.sh person --output-dir custom_vlpart_results

    # Use predefined vocabulary (avoids OpenAI API calls)
    ./run_vlpart.sh person --predefined-vocab "person head" "person arm" "person leg" "person torso"

OUTPUT STRUCTURE:
    vlpart_output_<supercategory>/     # VLPart segmentation results
    ├── processed_data/             # Pickled intermediate data (complex objects)
    ├── annotations/                   # JSON annotation files (human-readable)
    ├── masks/                         # Reference and target masks
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
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --min-area-ratio)
            MIN_AREA_RATIO="$2"
            shift 2
            ;;
        --max-area-ratio)
            MAX_AREA_RATIO="$2"
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
        --predefined-vocab)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                PREDEFINED_VOCAB="$PREDEFINED_VOCAB $1"
                shift
            done
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
            if [[ -z "$SUPERCATEGORY" ]]; then
                SUPERCATEGORY="$1"
            else
                print_error "Multiple supercategories specified: $SUPERCATEGORY and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SUPERCATEGORY" ]]; then
    print_error "Supercategory is required"
    show_help
    exit 1
fi

# Set output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="vlpart_output_${SUPERCATEGORY}"
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
    if [[ ! -f "batch_vlpart_segmentation.py" ]]; then
        print_error "batch_vlpart_segmentation.py not found in current directory"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Print configuration
print_configuration() {
    print_header "VLPART SEGMENTATION CONFIGURATION"
    echo "Supercategory:      $SUPERCATEGORY"
    echo "Artifact types:     $ARTIFACT_TYPES"
    echo "Max images:         ${MAX_IMAGES:-unlimited}"
    echo "Min area ratio:     $MIN_AREA_RATIO"
    echo "Max area ratio:     $MAX_AREA_RATIO"
    echo "Device:             $DEVICE"
    echo "Output directory:   $OUTPUT_DIR"
    echo "Resume:             $RESUME"
    echo "Predefined vocab:   ${PREDEFINED_VOCAB:-using OpenAI}"
    echo ""
}

# Run VLPart segmentation
run_vlpart_segmentation() {
    print_header "VLPART SEGMENTATION PROCESSING"
    
    # Build VLPart command
    vlpart_cmd="python batch_vlpart_segmentation.py $SUPERCATEGORY"
    vlpart_cmd="$vlpart_cmd --artifact-types $ARTIFACT_TYPES"
    vlpart_cmd="$vlpart_cmd --device $DEVICE"
    vlpart_cmd="$vlpart_cmd --min-area-ratio $MIN_AREA_RATIO"
    vlpart_cmd="$vlpart_cmd --max-area-ratio $MAX_AREA_RATIO"
    vlpart_cmd="$vlpart_cmd --output-dir $OUTPUT_DIR"
    
    if [[ -n "$MAX_IMAGES" ]]; then
        vlpart_cmd="$vlpart_cmd --max-images $MAX_IMAGES"
    fi
    
    if [[ "$RESUME" == true ]]; then
        vlpart_cmd="$vlpart_cmd --resume"
    fi
    
    if [[ -n "$PREDEFINED_VOCAB" ]]; then
        vlpart_cmd="$vlpart_cmd --predefined-vocab$PREDEFINED_VOCAB"
    fi
    
    print_info "Running VLPart segmentation..."
    print_info "Command: $vlpart_cmd"
    echo ""
    
    # Run VLPart segmentation
    start_time=$(date +%s)
    if $vlpart_cmd; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        print_success "VLPart segmentation completed in ${elapsed}s"
    else
        print_error "VLPart segmentation failed"
        exit 1
    fi
    
    # Verify VLPart output
    if [[ ! -d "$OUTPUT_DIR/processed_data" ]]; then
        print_error "VLPart output directory not created: $OUTPUT_DIR/processed_data"
        exit 1
    fi
    
    intermediate_count=$(find "$OUTPUT_DIR/processed_data" -name "image_*.pkl" | wc -l)
    print_info "Generated $intermediate_count intermediate data files"
    
    if [[ $intermediate_count -eq 0 ]]; then
        print_warning "No intermediate data files generated - this might indicate an issue"
    fi
}

# Print final summary
print_summary() {
    print_header "VLPART SEGMENTATION SUMMARY"
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        intermediate_count=$(find "$OUTPUT_DIR/processed_data" -name "image_*.pkl" 2>/dev/null | wc -l)
        annotations_count=$(find "$OUTPUT_DIR/annotations" -name "image_*_annotations.json" 2>/dev/null | wc -l)
        mask_count=$(find "$OUTPUT_DIR/masks" -name "*.png" 2>/dev/null | wc -l)
        echo "VLPart Results:"
        echo "  Output directory:     $OUTPUT_DIR"
        echo "  Intermediate files:   $intermediate_count"
        echo "  Annotation files:     $annotations_count"
        echo "  Mask files:           $mask_count"
        echo ""
    fi
    
    # Log files
    echo "Log Files:"
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        echo "  VLPart logs:          $OUTPUT_DIR/logs/"
    fi
    echo ""
    
    print_success "VLPart segmentation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  - Review the generated masks and intermediate data"
    echo "  - Check the log files for detailed processing information"
    echo "  - Use the results with run_flux.sh for artifact generation"
    echo "  - Command: ./run_flux.sh $OUTPUT_DIR --artifact-types \"$ARTIFACT_TYPES\""
}

# Cleanup function for interrupted execution
cleanup() {
    print_warning "VLPart processing interrupted"
    print_info "Partial results may be available in output directory: $OUTPUT_DIR"
    print_info "Use --resume to continue from where you left off"
    exit 1
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_header "VLPART SEGMENTATION PIPELINE"
    echo "Starting VLPart processing for supercategory: $SUPERCATEGORY"
    echo "Timestamp: $(date)"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Print configuration
    print_configuration
    
    # Record start time
    start_time=$(date +%s)
    
    # Run VLPart segmentation
    run_vlpart_segmentation
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