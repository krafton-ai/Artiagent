# Image Artifacts Pipeline

A modular pipeline for generating synthetic image artifacts using GSAM (Grounded Segment Anything Model) for part detection and FLUX diffusion model for artifact generation.

## Overview

This pipeline provides clean, modular Python components for the two-stage artifact generation workflow:

```
📦 pipeline/
├── 📄 __init__.py              # Package initialization
├── 📄 data_loader.py           # Dataset handling (COCO, ImageNet, Custom)
├── 📄 gsam_detector.py         # GSAM model integration (GroundingDINO + SAM)
├── 📄 instance_processor.py    # Instance filtering and bbox operations
├── 📄 flux_generator.py        # FLUX model operations
├── 📄 visualization.py         # Image visualization utilities
├── 📄 prompts.py               # OpenAI API utilities for vocabulary generation
└── 📄 README.md               # This file
```

## Architecture

```
┌─────────────────┐
│  COCO/ImageNet  │
│  Custom Dataset │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │
│  (data_loader)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  GSAM Detector  │◄────┤  OpenAI API  │
│  (GroundingDINO │     │  (Vocabulary)│
│   + SAM/SAM-HQ) │     └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Instance Processor│
│ (Filter, Sample,│
│  Create Patches)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FLUX Generator  │
│ (Diffusion with │
│ Patch Guidance) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualizer     │
│ (Results/Masks) │
└─────────────────┘
```

## Components

### 1. Data Loaders

Handles dataset loading and image sampling for COCO, ImageNet, and custom directories.

```python
from pipeline import COCODataLoader, ImageNetDataLoader, CustomDirectoryDataLoader

# COCO Dataset
coco_loader = COCODataLoader(
    dataset_path="/path/to/coco/annotations",
    image_path="/path/to/coco/images"
)
cat_ids = coco_loader.get_category_ids(['person'])
img_info, img_array, caption = coco_loader.sample_image_by_category(cat_ids)

# ImageNet Dataset
imagenet_loader = ImageNetDataLoader(
    dataset_path="/path/to/imagenet",
    split="train"
)

# Custom Directory
custom_loader = CustomDirectoryDataLoader(
    directory_path="/path/to/images"
)
```

### 2. GSAMDetector

Integrates GroundingDINO and SAM/SAM-HQ for part detection with OpenAI-generated vocabulary.

```python
from pipeline import GSAMDetector

detector = GSAMDetector(
    grounding_config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_checkpoint="weight/groundingdino_swint_ogc.pth",
    sam_checkpoint="weight/sam_vit_h_4b8939.pth",
    sam_version="vit_h",
    use_sam_hq=False,
    box_threshold=0.3,
    text_threshold=0.25,
    device='cuda'
)

# Detect parts in image
entity_predictions, subentity_predictions, vis_output = detector.detect_parts(
    image=img_array,
    entities=['person'],
    subentities=['person head', 'person arm', 'person leg'],
    entity_subentity_mapping={'person': ['person head', 'person arm', 'person leg']},
    min_area_ratio=0.005,
    max_area_ratio=0.5
)
```

### 3. InstanceProcessor

Handles instance filtering, sampling, bounding box operations, and patch annotation creation.

```python
from pipeline import InstanceProcessor

# Sample instance by confidence score
sampled_instance = InstanceProcessor.sample_instance_by_score(
    predictions, 
    min_area_ratio=0.01, 
    max_area_ratio=0.5
)

# Generate bbox suggestions for addition artifacts
suggested_bbox = InstanceProcessor.generate_bbox_suggestion(
    predictions=predictions,
    reference_bbox=reference_bbox,
    class_name=class_name,
    vocab=vocab,
    max_ref_overlap=0.3,
    min_entity_overlap=0.1
)

# Create annotation with patch indices
annotation_data = InstanceProcessor.create_annotation_dict(
    instance=sampled_instance,
    img_shape=image.shape,
    artifact_type='distortion',
    patch_size=16
)
```

### 4. FluxGenerator

Manages FLUX diffusion model operations for artifact generation with patch-based guidance.

```python
from pipeline import FluxGenerator, FluxConfig

# Configure FLUX model
config = FluxConfig(
    name='flux-dev',
    guidance=5.0,
    num_steps=25,
    pe_step=0.5,  # Position encoding step
    seed=42
)

# Artifact-type-specific PE steps
config_advanced = FluxConfig(
    name='flux-dev',
    guidance=5.0,
    num_steps=25,
    pe_step={
        'addition': 0.3,
        'removal': 0.3,
        'distortion': 0.5
    },
    seed=42
)

generator = FluxGenerator(device='cuda', config=config_advanced)

# Generate artifact image
generated_image = generator.generate_with_artifacts(
    source_prompt="a photo of a person",
    target_prompt="a photo of a person",
    bbox=target_bbox,
    bbox_ref=reference_bbox,
    artifact_type='distortion',
    source_img=image
)
```

### 5. ImageVisualizer

Provides visualization utilities for debugging and quality assurance.

```python
from pipeline import ImageVisualizer

visualizer = ImageVisualizer()

# Show single image with caption
visualizer.show_image(image, caption, title="Original", base_dir="output/")

# Show comparison
visualizer.show_comparison(
    original_image, 
    generated_image, 
    artifact_data,
    caption="Distortion Artifact",
    base_dir="output/"
)

# Show bounding box overlay
visualizer.show_bbox_overlay(
    image, 
    target_bbox,
    base_dir="output/",
    filename="bbox_overlay.png"
)

# Show patch masks
visualizer.show_patch_masks(
    image,
    reference_patches,
    target_patches,
    base_dir="output/"
)
```

## Artifact Types

The pipeline supports three types of image artifacts:

### 1. Distortion
Modifies the appearance of existing parts while keeping them in place.
- Uses reference patches to guide where distortion is applied
- Configurable distortion kernels (jitter, swirl, voronoi)

### 2. Removal
Removes detected parts from images naturally.
- Uses reference patches to identify removal areas
- FLUX inpainting fills removed areas contextually

### 3. Addition
Adds new instances of detected parts in suitable locations.
- Uses reference patches as source templates
- Generates target patches using IoU-based intelligent placement
- Maintains visual consistency with surrounding context

## Configuration

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `box_threshold` | `0.3` | Detection confidence threshold |
| `text_threshold` | `0.25` | Text-image matching threshold |
| `min_area_ratio` | `0.005` | Minimum part size (0.5% of image) |
| `max_area_ratio` | `0.5` | Maximum part size (50% of image) |
| `nms_threshold` | `0.5` | Non-maximum suppression threshold |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guidance` | `5.0` | Guidance scale for FLUX |
| `num_steps` | `25` | Number of diffusion steps |
| `pe_step` | `0.5` | Position encoding step size |
| `inject` | `25` | Injection step in diffusion |
| `seed` | `42` | Random seed for reproducibility |

## Dependencies

- `torch` - PyTorch for deep learning
- `openai` - For vocabulary generation
- `pycocotools` - For COCO dataset handling
- `supervision` - For detection utilities
- `groundingdino` - For grounded detection
- `segment_anything` - For segmentation
- `matplotlib` - For visualization
- `PIL` - For image processing
- `numpy` - For numerical operations

## Installation

1. Install GroundingDINO and SAM following their respective installation guides
2. Download model weights and place them in `src/weight/` directory
3. Set up OpenAI API key: `export OPENAI_API_KEY='your-key'`
4. Install required dependencies

See the main README for detailed installation instructions.

## Usage

See `batch_gsam_segmentation.py` and `batch_flux_generation.py` for complete batch processing examples.

## Performance Tips

1. **GPU Memory**: Use SAM `vit_b` for limited GPU memory, `vit_h` for best quality
2. **Filtering**: Adjust area ratios to balance quality vs. quantity
3. **Speed**: Lower `num_steps` (15-20) for faster generation
4. **Quality**: Higher `num_steps` (25-35) for better results

## Troubleshooting

### Common Issues

1. **GSAM setup issues**: Ensure GroundingDINO and SAM weights are downloaded
2. **OpenAI API errors**: Check API key and rate limits
3. **COCO dataset errors**: Verify dataset paths and structure
4. **GPU memory issues**: Use smaller SAM model or reduce batch size

## License

This pipeline integrates multiple open-source components, each with their own licenses. See the main repository LICENSE and model_licenses/ directory for details.
