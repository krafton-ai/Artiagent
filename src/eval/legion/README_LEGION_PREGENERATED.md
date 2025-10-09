# LEGION Evaluation with Pre-generated Responses

This solution allows running LEGION evaluations by pre-generating responses in the `legion1.4.7` environment and then running evaluations in the `lfac` environment.

## Problem

The LEGION model and evaluation dependencies are incompatible:
- `legion1.4.7` environment: Has LEGION model but lacks compatible transformers for eval scripts
- `lfac` environment: Has evaluation scripts but lacks LEGION model dependencies

## Solution

1. **Pre-generate LEGION responses** in `legion1.4.7` environment
2. **Run evaluations using pre-generated responses** in `lfac` environment

## Quick Start

### Step 1: Generate LEGION Responses

```bash
# Activate LEGION environment
conda activate legion1.4.7

# Navigate to eval directory
cd /home/jhpark/image-artifacts/src/eval

# Generate all responses (this may take a while)
bash generate_all_legion_responses.sh
```

### Step 2: Run Evaluations

```bash
# Activate evaluation environment  
conda activate lfac

# Navigate to eval directory
cd /home/jhpark/image-artifacts/src/eval

# Run all evaluations
bash run_legion_evaluation_with_pregenerated.sh
```

## Manual Usage

### Generate Responses for Specific Datasets

```bash
# In legion1.4.7 environment
python generate_legion_responses.py \
    --datasets synthscars synartifact \
    --output_dir ./legion_responses \
    --max_samples 50
```

### Run Individual Evaluations

```bash
# In lfac environment
python eval_with_pregenerated.py --model legion --dataset synthscars --type localization
python eval_with_pregenerated.py --model legion --dataset loki --type explanation
```

## File Structure

```
src/eval/
├── generate_legion_responses.py          # Pre-generation script (run in legion1.4.7)
├── eval_with_pregenerated.py            # Patched evaluation script (run in lfac)
├── mock_legion.py                        # Mock LEGION model implementation
├── generate_all_legion_responses.sh     # Convenience script for generation
├── run_legion_evaluation_with_pregenerated.sh  # Convenience script for evaluation
├── legion_responses/                     # Directory for pre-generated responses
│   ├── synthscars_responses.pkl
│   ├── synartifact_responses.pkl
│   ├── loki_responses.pkl
│   └── richhf_responses.pkl
└── README_LEGION_PREGENERATED.md        # This file
```

## How It Works

### 1. Pre-generation Phase (`legion1.4.7` environment)

- `generate_legion_responses.py` loads the real LEGION model
- Processes all images from specified datasets
- Generates responses in the format expected by evaluation scripts
- Saves responses as pickle files with image filename as key

### 2. Evaluation Phase (`lfac` environment)

- `models.py` automatically detects available pre-generated responses
- `LegionEval` class switches to `MockLegionEval` when responses are available
- `MockLegionEval` loads pre-generated responses and returns them during inference
- Thread-local context passes image paths from evaluation loops to mock model
- Evaluation scripts run normally, unaware they're using pre-generated responses

## Supported Evaluations

All evaluations from `legion_evaluation.sh` are supported:

### WSOL Evaluation (Localization 1)
- `wsol_eval.py` with visualization integration

### LEGION-like Evaluation (Localization 2) 
- `legion_eval.py` with LEGION-specific metrics

### Bbox-map Evaluation (Localization 3)
- `eval.py` with bounding box evaluation  

### Explanation Evaluation
- Text-based artifact explanation evaluation

## Response Format

Pre-generated responses match the LEGION model output format:

```python
{
    "heatmap": torch.Tensor,      # Segmentation mask as tensor
    "explanation": str,           # Text explanation of artifacts
    # Optional error field if generation failed
    "error": str                  
}
```

## Troubleshooting

### No Pre-generated Responses Found
```
⚠️ No pre-generated responses found at /path/to/responses.pkl
```
- Run the generation script first in `legion1.4.7` environment
- Check the output directory path

### Import Errors in legion1.4.7
```
ImportError: cannot import name 'Qwen2_5_VLForConditionalGeneration'
```
- This is expected - only run generation script in `legion1.4.7` 
- The generation script has minimal imports to avoid this issue

### Missing Images During Generation
```
⚠️ Image not found: /path/to/image.jpg
```
- Check dataset paths in `generate_legion_responses.py`
- Ensure dataset directory structure is correct

### Context Path Issues
```
⚠️ No current image path available in context
```
- Use `eval_with_pregenerated.py` instead of original scripts
- This passes image paths correctly to the mock model

## Performance

- **Generation**: ~30-60 seconds per image depending on LEGION model complexity
- **Evaluation**: Near-instant response lookup from pre-generated files
- **Storage**: ~1-10MB per dataset depending on mask complexity

## Extending to Other Models

The same approach can be used for other models with compatibility issues:

1. Create pre-generation script for the problematic model
2. Create mock class that loads pre-generated responses  
3. Update model class to auto-switch to mock when responses available
4. Use thread-local context for passing additional information

## Datasets

Supports all datasets in the evaluation pipeline:

- **SynthScars**: Synthetic scarring artifacts
- **SynArtifact**: Synthetic visual artifacts  
- **LOKI**: Real-world artifact dataset
- **RichHF**: Rich human feedback dataset

Dataset paths are automatically detected based on standard locations.
