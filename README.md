# Image Artifacts Generation Pipeline

This repository contains a two-stage pipeline for generating image artifacts using GSAM (Grounded Segment Anything Model) segmentation and FLUX diffusion models.

## Pipeline Overview

```
Input Images → GSAM Processing → FLUX Generation → Artifact Images
```

The pipeline consists of two main stages:

1. **GSAM Segmentation** (`batch_gsam_segmentation.py`): Processes images to detect and segment parts, creating patch-based annotations
2. **FLUX Artifact Generation** (`batch_flux_generation.py`): Uses the segmentation results to generate artifacts with FLUX diffusion model

## Stage 1: GSAM Segmentation Processing

### Overview
The GSAM processing stage analyzes input images to identify and segment object parts, creating detailed annotations that guide the artifact generation process.

### Key Features
- **Multi-dataset Support**: Works with COCO, ImageNet, and custom directory datasets
- **Intelligent Part Detection**: Uses GroundingDINO + SAM for precise part segmentation
- **Patch-based Annotations**: Creates 16×16 pixel patch indices for fine-grained control
- **Artifact Type Support**: Handles distortion, removal, and addition artifacts
- **Smart Filtering**: Uses area ratios and overlap thresholds for quality control
- **Progress Tracking**: Resumable processing with detailed progress logs

### Processing Workflow

1. **Dataset Loading**
   - Loads images from specified dataset (COCO/ImageNet/Custom)
   - Supports category-based filtering and image limits

2. **Part Vocabulary Generation**
   - Uses OpenAI API to generate contextual part vocabulary
   - Adapts vocabulary based on target artifact type
   - Example: For "person" → generates "person head", "person arm", "person leg"

3. **Part Detection & Segmentation**
   - GroundingDINO detects parts based on generated vocabulary
   - SAM/SAM-HQ creates precise segmentation masks
   - Filters detections by confidence thresholds

4. **Target Part Sampling**
   - Selects appropriate parts based on area ratios (default: 0.5%-50% of image)
   - Ensures parts are suitable for the target artifact type

5. **Patch Annotation Creation**
   - Converts segmentation masks to 16×16 pixel patch indices
   - Creates reference patches (original part location)
   - For addition artifacts: generates smart target patches using IoU-based placement
   - For removal/distortion: uses reference patches only

6. **Data Unification & Storage**
   - Saves all processing results in unified pickle format
   - Includes image data, annotations, masks, and patch indices
   - Creates visualization outputs for quality assurance

### Usage Example

```bash
# Process custom dataset with animal images
python batch_gsam_segmentation.py animal \
    --dataset custom \
    --dataset-path /path/to/animal/images \
    --artifact-types distortion removal addition \
    --max-images 100 \
    --output-dir ../exps/your-exp-name

# Process COCO dataset
python batch_gsam_segmentation.py person vehicle \
    --dataset coco \
    --dataset-path /path/to/coco/annotations \
    --image-path /path/to/coco/images \
    --min-area-ratio 0.01 \
    --max-area-ratio 0.4
```

```
./run_gsam.sh  animal --max-images 10 --output-dir ../exps/your-exp-name
```

### Output Structure
```
gsam_output_animals/
├── processed_data/
│   ├── image_12345.pkl        # Unified data for each image
│   └── image_67890.pkl
├── logs/
│   └── gsam_processing_*.log  # Processing logs
├── processing_progress.json   # Progress tracking
└── [image_name]/
    ├── 01_original_image.png
    ├── 02_detection_results.png
    └── 03_patch_masks_*.png
```

## Stage 2: FLUX Artifact Generation

### Overview
The FLUX generation stage reads the GSAM processing results and generates three types of image artifacts using patch-based guidance with the FLUX diffusion model.

### Key Features
- **Patch-guided Generation**: Uses precise patch indices from GSAM processing
- **Multiple Artifact Types**: Supports distortion, removal, and addition artifacts
- **Flexible Parameters**: Configurable guidance, steps, and injection timing
- **Visualization Pipeline**: Creates before/after comparisons and patch overlays
- **Resumable Processing**: Continues from previous runs with progress tracking

### Artifact Types

1. **Distortion Artifacts**
   - Applies visual distortions to detected parts
   - Uses reference patches to guide distortion placement
   - Configurable distortion kernels (jitter, swirl, voronoi)

2. **Removal Artifacts**
   - Removes detected parts from images
   - Uses reference patches to identify removal areas
   - Fills removed areas naturally using FLUX inpainting

3. **Addition Artifacts**
   - Adds new instances of detected parts
   - Uses reference patches as source templates
   - Uses target patches for intelligent placement
   - Maintains visual consistency and context

### Processing Workflow

1. **Data Loading**
   - Reads unified processing results from GSAM stage
   - Validates patch annotations and artifact data

2. **FLUX Model Initialization**
   - Loads FLUX-dev model with specified configuration
   - Sets up guidance, steps, and injection parameters

3. **Artifact Generation**
   - For each valid annotation:
     - Extracts reference and target patch indices
     - Applies FLUX model with patch-based guidance
     - Generates artifact image using diffusion process

4. **Visualization Creation**
   - Creates comparison images (original vs. generated)
   - Overlays patch visualizations showing guided areas
   - Saves high-quality output images

5. **Progress Tracking**
   - Logs successful/failed generations per artifact type
   - Tracks processing times and success rates
   - Enables resumable processing

### Usage Example

```bash
# Generate artifacts from GSAM processing results
python batch_flux_generation.py gsam_output_animals \
    --artifact-types distortion removal addition \
    --inject 25 \
    --guidance 5.0 \
    --num-steps 25 \
    --resume

# Custom FLUX parameters
python batch_flux_generation.py gsam_output_animals \
    --pe-step-addition 0.3 \
    --pe-step-removal 0.3 \
    --pe-step-distortion 0.5 \
    --seed 42
```

```
./run_flux.sh ../exps/gsam-exp-name --output-dir ../exps/your-exp-name
```

### Output Structure
```
flux_output_animals/
├── logs/
│   └── flux_generation_*.log
├── flux_progress.json
└── [image_name]/
    ├── 01_original_image.png      # From GSAM stage
    ├── 02_detection_results.png   # From GSAM stage
    ├── 04_comparison_distortion.png
    ├── 04_comparison_removal.png
    ├── 04_comparison_addition.png
    ├── 06_patches_distortion.png  # Patch visualizations
    ├── 06_patches_removal.png
    ├── 06_patches_addition.png
    └── 07_injected_image_*.png    # Final generated images
```

## Configuration & Requirements
make **two** separate conda environment, following the installation guides
### Grounding-SAM
Follow the installation guide in [Grounding SAM installation guide](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#install-without-docker) to set up the Grounding DINO environment with SAM integration.

Download the pretrained weights, and save it is src/weight directory.
```
cd src/weight

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### RF-Solver-Edit

Follow the installation guide in [RF-Solver-Edit installation guide](https://github.com/wangjiangshan0725/RF-Solver-Edit/tree/main/FLUX_Image_Edit#%EF%B8%8F-code-setup).

### Open api key
Save the OPENAI_API_KEY in .env file.

### Key Parameters

#### GSAM Processing Parameters (`batch_gsam_segmentation.py`)

**Required Arguments:**
- `categories`: Category names to process (e.g., `person`, `animal`, `vehicle`)

**Dataset Configuration:**
- `--dataset`: Dataset type - `coco`, `imagenet`, or `custom` (default: `custom`)
- `--dataset-path`: Path to dataset root directory
- `--image-path`: Path to images (COCO only)
- `--imagenet-split`: ImageNet split - `train` or `val` (default: `train`)

**Processing Control:**
- `--artifact-types`: Artifact types to generate (default: `distortion removal addition`)
- `--max-images`: Maximum number of images to process (default: unlimited)
- `--resume`: Resume from previous run (flag)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--output-dir`: Output directory (default: `../gsam_output_eval_animals`)

**Part Detection & Filtering:**
- `--min-area-ratio`: Minimum part size as ratio of image area (default: `0.005` = 0.5%)
- `--max-area-ratio`: Maximum part size as ratio of image area (default: `0.5` = 50%)
- `--max-ref-overlap`: Maximum overlap with reference bbox for addition artifacts (default: `0.3`)
- `--min-entity-overlap`: Minimum overlap with entity bbox for addition artifacts (default: `0.1`)
- `--predefined-vocab`: Pre-defined vocabulary list to skip OpenAI API calls

**GroundingDINO Configuration:**
- `--grounding-config`: Path to GroundingDINO config file (default: `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`)
- `--grounding-checkpoint`: Path to GroundingDINO checkpoint (default: `weight/groundingdino_swint_ogc.pth`)
- `--box-threshold`: Detection confidence threshold (default: `0.3`)
- `--text-threshold`: Text-image matching threshold (default: `0.25`)
- `--nms-threshold`: Non-maximum suppression threshold (default: `0.5`)
- `--bert-base-uncased-path`: Path to BERT model (optional)

**SAM Configuration:**
- `--sam-version`: SAM model version - `vit_b`, `vit_l`, or `vit_h` (default: `vit_h`)
- `--sam-checkpoint`: Path to SAM checkpoint (default: `weight/sam_vit_h_4b8939.pth`)
- `--sam-hq-checkpoint`: Path to SAM-HQ checkpoint (optional)
- `--use-sam-hq`: Use SAM-HQ for higher quality (flag)

**Distortion Options:**
- `--distortion-kernel`: Distortion type - `none`, `jitter`, `swirl`, or `voronoi` (default: `none`)

#### FLUX Generation Parameters (`batch_flux_generation.py`)

**Required Arguments:**
- `segmentation_output_dir`: Directory containing GSAM processing results

**Processing Control:**
- `--artifact-types`: Artifact types to generate (default: `distortion removal addition`)
- `--resume`: Resume from previous run (flag)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--output-dir`: Output directory (default: `flux_output_{supercategory}`)

**FLUX Model Parameters:**
- `--inject`: Injection step in diffusion process (default: `25`)
- `--guidance`: Guidance scale for generation (default: `5.0`)
- `--num-steps`: Total diffusion steps (default: `25`)
- `--seed`: Random seed for reproducibility (default: `42`)

**Position Encoding Steps (Per Artifact Type):**
- `--pe-step-addition`: PE step for addition artifacts (default: `0.3`)
- `--pe-step-removal`: PE step for removal artifacts (default: `0.3`)
- `--pe-step-distortion`: PE step for distortion artifacts (default: `0.5`)

#### Parameter Tuning Guidelines

**Area Ratio Filtering:**
- **Small parts (0.005-0.05)**: Good for detailed features like eyes, buttons, small decorations
- **Medium parts (0.05-0.2)**: Ideal for body parts, faces, wheels, significant object components
- **Large parts (0.2-0.5)**: Suitable for major object regions, backgrounds, dominant features

**Detection Thresholds:**
- **Low thresholds (0.1-0.3)**: More detections, may include false positives
- **High thresholds (0.3-0.5)**: Fewer, more confident detections
- **Text threshold**: Controls text-image matching strictness

**FLUX Generation:**
- **Low inject steps (15-20)**: More creative, less constrained generation
- **High inject steps (25-30)**: More controlled, faithful to original
- **Low guidance (3.0-5.0)**: More natural, diverse outputs
- **High guidance (7.5-10.0)**: Stronger adherence to prompts
- **PE steps**: Control strength of positional encoding per artifact type

**Performance vs Quality Trade-offs:**
- **Faster processing**: Lower num_steps (15-20), higher inject steps (25-30)
- **Higher quality**: More num_steps (25-35), lower inject steps (15-20)
- **Memory usage**: vit_h uses most memory, vit_b uses least

#### Common Parameter Combinations

**High Quality Processing:**
```bash
# GSAM stage
--min-area-ratio 0.01 --max-area-ratio 0.4 --box-threshold 0.35 --use-sam-hq

# FLUX stage  
--inject 20 --guidance 7.5 --num-steps 30 --pe-step-distortion 0.7
```

**Fast Processing:**
```bash
# GSAM stage
--min-area-ratio 0.02 --max-area-ratio 0.3 --box-threshold 0.25 --sam-version vit_b

# FLUX stage
--inject 25 --guidance 5.0 --num-steps 20 --pe-step-addition 0.2
```

**Creative Generation:**
```bash
# FLUX stage
--inject 15 --guidance 3.0 --num-steps 25 --pe-step-removal 0.4
```