#!/bin/bash

# =============================================================================
# GSAM (Grounded SAM) Segmentation Pipeline Script
# =============================================================================
# This script runs GSAM segmentation batch processing to generate part 
# segmentation masks and annotations using GroundingDINO + SAM.
#
# Usage: ./run_gsam.sh [supercategory] [options]
# Example: ./run_gsam.sh person --max-images 100 --device cuda
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
SUPERCATEGORIES=()  # Array to hold multiple supercategories
# ARTIFACT_TYPES="addition removal distortion"
ARTIFACT_TYPES="addition removal distortion fusion"
DISTORTION_KERNEL="random jitter strip"
RANDOM_DISTORTION=true
MAX_IMAGES=""
MIN_AREA_RATIO="0.005"
MAX_AREA_RATIO="0.5"
DEVICE="cuda"
RESUME=false
OUTPUT_DIR=""
PREDEFINED_VOCAB=""
SEED=""

# Dataset selection defaults (uncomment and set paths as needed)
# DATASET="custom"
# DATASET_PATH="/path/to/your/dataset"
# IMAGE_PATH="/path/to/your/images"
# IMAGENET_SPLIT="train"

# DATASET="coco"
# DATASET_PATH="/path/to/coco/annotations"
# IMAGE_PATH="/path/to/coco/train2017"
# IMAGENET_SPLIT="train"

# GSAM-specific defaults
GROUNDING_CONFIG=""
GROUNDING_CHECKPOINT=""
SAM_VERSION="vit_h"
SAM_CHECKPOINT=""
SAM_HQ_CHECKPOINT=""
USE_SAM_HQ=false
BOX_THRESHOLD="0.3"
TEXT_THRESHOLD="0.25"
BERT_BASE_UNCASED_PATH=""

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
GSAM (Grounded SAM) Segmentation Pipeline

USAGE:
    ./run_gsam.sh <supercategory1> [supercategory2] ... [OPTIONS]

ARGUMENTS:
    supercategory...        One or more COCO supercategories to process (person, animal, vehicle, etc.)

BASIC OPTIONS:
    --artifact-types LIST   Artifact types to generate (default: "distortion removal addition")
    --distortion-kernel TYPE Distortion kernel type: random, jitter, strip (default: random)
    --random-distortion     Randomly sample distortion kernel for each image (random, jitter, strip)
    --max-images N          Maximum number of images to process
    --min-area-ratio FLOAT  Minimum area ratio for part filtering (default: 0.05)
    --max-area-ratio FLOAT  Maximum area ratio for part filtering (default: 0.5)
    --device DEVICE         Device to use (default: cuda)
    --output-dir DIR        Output directory (default: gsam_output_<supercategory>)
    --resume                Resume processing from previous run
    --predefined-vocab LIST Pre-defined vocabulary (e.g., --predefined-vocab "person head" "person arm")
    --seed SEED             Random seed for reproducible results

DATASET OPTIONS:
    --dataset TYPE          Dataset type: coco, imagenet, custom (default: custom)
    --dataset-path PATH     Path to dataset root directory
    --image-path PATH       Path to images (COCO only, for ImageNet use --imagenet-split)
    --imagenet-split SPLIT  ImageNet split: train, val (default: train)

GSAM-SPECIFIC OPTIONS:
    --grounding-config PATH     Path to GroundingDINO config file
    --grounding-checkpoint PATH Path to GroundingDINO checkpoint
    --sam-version VERSION       SAM model version: vit_b, vit_l, vit_h (default: vit_h)
    --sam-checkpoint PATH       Path to SAM checkpoint
    --sam-hq-checkpoint PATH    Path to SAM-HQ checkpoint
    --use-sam-hq                Use SAM-HQ instead of regular SAM
    --box-threshold FLOAT       Box threshold for GroundingDINO (default: 0.3)
    --text-threshold FLOAT      Text threshold for GroundingDINO (default: 0.25)
    --bert-base-uncased-path PATH Path to BERT base uncased model

    --help                      Show this help message

EXAMPLES:
    # Basic usage (single supercategory)
    ./run_gsam.sh person

    # Multiple supercategories
    ./run_gsam.sh person animal vehicle

    # Process COCO dataset
    ./run_gsam.sh person --dataset coco --dataset-path /path/to/coco/annotations --image-path /path/to/coco/images

    # Process ImageNet dataset
    ./run_gsam.sh animal --dataset imagenet --dataset-path /path/to/imagenet --imagenet-split train

    # Custom dataset with specific path
    ./run_gsam.sh cat --dataset custom --dataset-path /path/to/custom/dataset

    # Process 100 images with custom filtering
    ./run_gsam.sh person --max-images 100 --min-area-ratio 0.02 --max-area-ratio 0.6

    # Only generate distortion artifacts
    ./run_gsam.sh animal --artifact-types "distortion"

    # Generate distortion artifacts with jitter kernel
    ./run_gsam.sh animal --artifact-types "distortion" --distortion-kernel jitter

    # Generate distortion artifacts with strip kernel
    ./run_gsam.sh animal --artifact-types "distortion" --distortion-kernel strip

    # Randomly sample distortion kernel for each image
    ./run_gsam.sh animal --artifact-types "distortion" --random-distortion

    # Resume interrupted processing
    ./run_gsam.sh person --resume

    # Custom output directory
    ./run_gsam.sh person --output-dir custom_gsam_results

    # Use predefined vocabulary (avoids OpenAI API calls)
    ./run_gsam.sh person --predefined-vocab "person head" "person arm" "person leg" "person torso"

    # Use SAM-HQ for better quality
    ./run_gsam.sh person --use-sam-hq --sam-hq-checkpoint /path/to/sam_hq.pth

    # Custom thresholds for detection
    ./run_gsam.sh person --box-threshold 0.35 --text-threshold 0.3

    # Full configuration example with COCO dataset
    ./run_gsam.sh person \
        --dataset coco \
        --dataset-path /path/to/coco/annotations \
        --image-path /path/to/coco/images \
        --grounding-checkpoint /models/groundingdino.pth \
        --sam-checkpoint /models/sam_vit_h.pth \
        --max-images 50 \
        --device cuda

OUTPUT STRUCTURE:
    gsam_output_<supercategory>/        # GSAM segmentation results
    ├── processed_data/                 # Pickled intermediate data (complex objects)
    ├── annotations/                    # JSON annotation files (human-readable)
    ├── masks/                          # Reference and target masks
    └── logs/                           # Processing logs

SETUP REQUIREMENTS:
    Before running, ensure you have:
    1. GroundingDINO and Segment Anything installed
    2. Model checkpoints downloaded
    3. OPENAI_API_KEY environment variable set

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --artifact-types)
            ARTIFACT_TYPES="$2"
            shift 2
            ;;
        --distortion-kernel)
            DISTORTION_KERNEL="$2"
            shift 2
            ;;
        --random-distortion)
            RANDOM_DISTORTION=true
            shift
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
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --image-path)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --imagenet-split)
            IMAGENET_SPLIT="$2"
            shift 2
            ;;
        --grounding-config)
            GROUNDING_CONFIG="$2"
            shift 2
            ;;
        --grounding-checkpoint)
            GROUNDING_CHECKPOINT="$2"
            shift 2
            ;;
        --sam-version)
            SAM_VERSION="$2"
            shift 2
            ;;
        --sam-checkpoint)
            SAM_CHECKPOINT="$2"
            shift 2
            ;;
        --sam-hq-checkpoint)
            SAM_HQ_CHECKPOINT="$2"
            shift 2
            ;;
        --use-sam-hq)
            USE_SAM_HQ=true
            shift
            ;;
        --box-threshold)
            BOX_THRESHOLD="$2"
            shift 2
            ;;
        --text-threshold)
            TEXT_THRESHOLD="$2"
            shift 2
            ;;
        --bert-base-uncased-path)
            BERT_BASE_UNCASED_PATH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -*)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            # Collect supercategories (all positional arguments that aren't options)
            SUPERCATEGORIES+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ ${#SUPERCATEGORIES[@]} -eq 0 ]]; then
    print_error "At least one supercategory is required"
    show_help
    exit 1
fi

# Validate dataset type
if [[ "$DATASET" != "coco" && "$DATASET" != "imagenet" && "$DATASET" != "custom" ]]; then
    print_error "Invalid dataset type: $DATASET. Must be one of: coco, imagenet, custom"
    exit 1
fi

# Validate distortion kernel type
if [[ "$DISTORTION_KERNEL" != "random" && "$DISTORTION_KERNEL" != "jitter" && "$DISTORTION_KERNEL" != "strip" ]]; then
    print_error "Invalid distortion kernel: $DISTORTION_KERNEL. Must be one of: random, jitter, strip"
    exit 1
fi

# Set output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    # Join supercategories with underscores for directory name
    SUPERCATEGORY_STRING=$(IFS=_; echo "${SUPERCATEGORIES[*]}")
    OUTPUT_DIR="gsam_output_${DATASET}_${SUPERCATEGORY_STRING}"
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
    if [[ ! -f "batch_gsam_segmentation.py" ]]; then
        print_error "batch_gsam_segmentation.py not found in current directory"
        exit 1
    fi
    
    # Check OpenAI API key
    if [[ -z "$OPENAI_API_KEY" ]]; then
        print_warning "OPENAI_API_KEY environment variable not set"
        print_info "Vocabulary generation will be limited without OpenAI API access"
    fi

    print_success "Dependencies check passed"
}

# Print configuration
print_configuration() {
    print_header "GSAM SEGMENTATION CONFIGURATION"
    echo "Supercategories:         ${SUPERCATEGORIES[*]}"
    echo "Artifact types:          $ARTIFACT_TYPES"
    if [[ "$RANDOM_DISTORTION" == true ]]; then
        echo "Distortion kernel:       random sampling (random, jitter, strip)"
    else
        echo "Distortion kernel:       $DISTORTION_KERNEL"
    fi
    echo "Max images:              ${MAX_IMAGES:-unlimited}"
    echo "Min area ratio:          $MIN_AREA_RATIO"
    echo "Max area ratio:          $MAX_AREA_RATIO"
    echo "Device:                  $DEVICE"
    echo "Output directory:        $OUTPUT_DIR"
    echo "Resume:                  $RESUME"
    echo "Predefined vocab:        ${PREDEFINED_VOCAB:-using OpenAI}"
    echo "Seed:                    ${SEED:-none}"
    echo ""
    echo "Dataset Configuration:"
    echo "  Dataset type:          $DATASET"
    echo "  Dataset path:          ${DATASET_PATH:-auto-detected}"
    echo "  Image path:            ${IMAGE_PATH:-auto-detected (COCO only)}"
    echo "  ImageNet split:        $IMAGENET_SPLIT"
    echo ""
    echo "GSAM Configuration:"
    echo "  Grounding config:      ${GROUNDING_CONFIG:-auto-detected}"
    echo "  Grounding checkpoint:  ${GROUNDING_CHECKPOINT:-auto-detected}"
    echo "  SAM version:           $SAM_VERSION"
    echo "  SAM checkpoint:        ${SAM_CHECKPOINT:-auto-detected}"
    echo "  SAM-HQ checkpoint:     ${SAM_HQ_CHECKPOINT:-not set}"
    echo "  Use SAM-HQ:            $USE_SAM_HQ"
    echo "  Box threshold:         $BOX_THRESHOLD"
    echo "  Text threshold:        $TEXT_THRESHOLD"
    echo "  BERT path:             ${BERT_BASE_UNCASED_PATH:-default}"
    echo "  Seed:                   ${SEED:-none}"
    echo ""
}

# Run GSAM segmentation
run_gsam_segmentation() {
    print_header "GSAM SEGMENTATION PROCESSING"
    
    # Process each supercategory
    for SUPERCATEGORY in "${SUPERCATEGORIES[@]}"; do
        print_info "Processing supercategory: $SUPERCATEGORY"
        echo ""
        
        # Build GSAM command for current supercategory
        gsam_cmd="python batch_gsam_segmentation.py $SUPERCATEGORY"
        gsam_cmd="$gsam_cmd --artifact-types $ARTIFACT_TYPES"
        gsam_cmd="$gsam_cmd --distortion-kernel $DISTORTION_KERNEL"
        gsam_cmd="$gsam_cmd --device $DEVICE"
        gsam_cmd="$gsam_cmd --min-area-ratio $MIN_AREA_RATIO"
        gsam_cmd="$gsam_cmd --max-area-ratio $MAX_AREA_RATIO"
        
        # Create supercategory-specific output directory if processing multiple categories
        if [[ ${#SUPERCATEGORIES[@]} -gt 1 ]]; then
            CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${SUPERCATEGORY}"
        else
            CURRENT_OUTPUT_DIR="$OUTPUT_DIR"
        fi
        gsam_cmd="$gsam_cmd --output-dir $CURRENT_OUTPUT_DIR"
        gsam_cmd="$gsam_cmd --dataset $DATASET"
        gsam_cmd="$gsam_cmd --sam-version $SAM_VERSION"
        gsam_cmd="$gsam_cmd --box-threshold $BOX_THRESHOLD"
        gsam_cmd="$gsam_cmd --text-threshold $TEXT_THRESHOLD"
        
        if [[ -n "$MAX_IMAGES" ]]; then
            gsam_cmd="$gsam_cmd --max-images $MAX_IMAGES"
        fi
        
        if [[ "$RESUME" == true ]]; then
            gsam_cmd="$gsam_cmd --resume"
        fi
        
        if [[ "$RANDOM_DISTORTION" == true ]]; then
            gsam_cmd="$gsam_cmd --random-distortion"
        fi
        
        if [[ -n "$PREDEFINED_VOCAB" ]]; then
            gsam_cmd="$gsam_cmd --predefined-vocab$PREDEFINED_VOCAB"
        fi
        
        if [[ -n "$GROUNDING_CONFIG" ]]; then
            gsam_cmd="$gsam_cmd --grounding-config $GROUNDING_CONFIG"
        fi
        
        if [[ -n "$GROUNDING_CHECKPOINT" ]]; then
            gsam_cmd="$gsam_cmd --grounding-checkpoint $GROUNDING_CHECKPOINT"
        fi
        
        if [[ -n "$SAM_CHECKPOINT" ]]; then
            gsam_cmd="$gsam_cmd --sam-checkpoint $SAM_CHECKPOINT"
        fi
        
        if [[ -n "$SAM_HQ_CHECKPOINT" ]]; then
            gsam_cmd="$gsam_cmd --sam-hq-checkpoint $SAM_HQ_CHECKPOINT"
        fi
        
        if [[ "$USE_SAM_HQ" == true ]]; then
            gsam_cmd="$gsam_cmd --use-sam-hq"
        fi
        
        if [[ -n "$BERT_BASE_UNCASED_PATH" ]]; then
            gsam_cmd="$gsam_cmd --bert-base-uncased-path $BERT_BASE_UNCASED_PATH"
        fi

        if [[ -n "$SEED" ]]; then
            gsam_cmd="$gsam_cmd --seed $SEED"
        fi
        
        # Add dataset-specific arguments
        if [[ -n "$DATASET_PATH" ]]; then
            gsam_cmd="$gsam_cmd --dataset-path $DATASET_PATH"
        fi
        
        if [[ -n "$IMAGE_PATH" ]]; then
            gsam_cmd="$gsam_cmd --image-path $IMAGE_PATH"
        fi
        
        if [[ "$DATASET" == "imagenet" ]]; then
            gsam_cmd="$gsam_cmd --imagenet-split $IMAGENET_SPLIT"
        fi
        
        print_info "Running GSAM segmentation for $SUPERCATEGORY..."
        print_info "Command: $gsam_cmd"
        echo ""
        
        # Run GSAM segmentation
        start_time=$(date +%s)
        if $gsam_cmd; then
            end_time=$(date +%s)
            elapsed=$((end_time - start_time))
            print_success "GSAM segmentation for $SUPERCATEGORY completed in ${elapsed}s"
        else
            print_error "GSAM segmentation for $SUPERCATEGORY failed"
            exit 1
        fi
        
        echo ""
    done
    
}

# Print final summary
print_summary() {
    print_header "GSAM SEGMENTATION SUMMARY"
    
    echo "GSAM Results:"
    for SUPERCATEGORY in "${SUPERCATEGORIES[@]}"; do
        if [[ ${#SUPERCATEGORIES[@]} -gt 1 ]]; then
            CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${SUPERCATEGORY}"
        else
            CURRENT_OUTPUT_DIR="$OUTPUT_DIR"
        fi
        
        if [[ -d "$CURRENT_OUTPUT_DIR" ]]; then
            intermediate_count=$(find "$CURRENT_OUTPUT_DIR/processed_data" -name "image_*.pkl" 2>/dev/null | wc -l)
            annotations_count=$(find "$CURRENT_OUTPUT_DIR/annotations" -name "image_*_annotations.json" 2>/dev/null | wc -l)
            mask_count=$(find "$CURRENT_OUTPUT_DIR/masks" -name "*.png" 2>/dev/null | wc -l)
            echo "  $SUPERCATEGORY:"
            echo "    Output directory:     $CURRENT_OUTPUT_DIR"
            echo "    Intermediate files:   $intermediate_count"
            echo "    Annotation files:     $annotations_count"
            echo "    Mask files:           $mask_count"
            echo ""
        fi
    done
    
    # Log files
    echo "Log Files:"
    for SUPERCATEGORY in "${SUPERCATEGORIES[@]}"; do
        if [[ ${#SUPERCATEGORIES[@]} -gt 1 ]]; then
            CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${SUPERCATEGORY}"
        else
            CURRENT_OUTPUT_DIR="$OUTPUT_DIR"
        fi
        if [[ -d "$CURRENT_OUTPUT_DIR/logs" ]]; then
            echo "  $SUPERCATEGORY logs:  $CURRENT_OUTPUT_DIR/logs/"
        fi
    done
    echo ""
    
    print_success "GSAM segmentation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  - Review the generated masks and intermediate data"
    echo "  - Check the log files for detailed processing information"
    echo "  - Use the results with run_flux.sh for artifact generation"
    if [[ ${#SUPERCATEGORIES[@]} -gt 1 ]]; then
        echo "  - Process each supercategory separately with run_flux.sh:"
        for SUPERCATEGORY in "${SUPERCATEGORIES[@]}"; do
            CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${SUPERCATEGORY}"
            echo "    ./run_flux.sh $CURRENT_OUTPUT_DIR --artifact-types \"$ARTIFACT_TYPES\""
        done
    else
        echo "  - Command: ./run_flux.sh $OUTPUT_DIR --artifact-types \"$ARTIFACT_TYPES\""
    fi
    if [[ "$RANDOM_DISTORTION" == true ]]; then
        echo "  - Note: Used random distortion kernels (random, jitter, strip) for distortion artifacts"
    elif [[ "$DISTORTION_KERNEL" != "random" ]]; then
        echo "  - Note: Used distortion kernel '$DISTORTION_KERNEL' for distortion artifacts"
    fi
    echo ""
    echo "Dataset processed: $DATASET"
    if [[ "$DATASET" == "coco" && -n "$DATASET_PATH" ]]; then
        echo "COCO annotations: $DATASET_PATH"
        if [[ -n "$IMAGE_PATH" ]]; then
            echo "COCO images: $IMAGE_PATH"
        fi
    elif [[ "$DATASET" == "imagenet" && -n "$DATASET_PATH" ]]; then
        echo "ImageNet path: $DATASET_PATH"
        echo "ImageNet split: $IMAGENET_SPLIT"
    elif [[ "$DATASET" == "custom" && -n "$DATASET_PATH" ]]; then
        echo "Custom dataset path: $DATASET_PATH"
    fi
}

# Cleanup function for interrupted execution
cleanup() {
    print_warning "GSAM processing interrupted"
    print_info "Partial results may be available in output directory: $OUTPUT_DIR"
    print_info "Use --resume to continue from where you left off"
    exit 1
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_header "GSAM SEGMENTATION PIPELINE"
    echo "Starting GSAM processing for supercategories: ${SUPERCATEGORIES[*]}"
    echo "Timestamp: $(date)"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Print configuration
    print_configuration
    
    # Record start time
    start_time=$(date +%s)
    
    # Run GSAM segmentation
    run_gsam_segmentation
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