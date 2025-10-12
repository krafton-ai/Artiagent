# eval_prompt_match.py Usage Guide

Unified evaluation script for artifact detection models across three tasks: **binary classification**, **localization**, and **explanation**.

## Quick Start

```bash
# Basic evaluation
python eval_prompt_match.py --model qwen --dataset ours

# With custom settings
python eval_prompt_match.py --model qwen --dataset ours --max-samples 100 --device cuda:0
```

## Command-Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `qwen`, `intern`, `gpt`, `gemini`, `pal`, `diff`, `legion` | `qwen` | Model type to evaluate |
| `--dataset` | `synthscars`, `synartifact`, `loki`, `richhf`, `ours`, `val` | `ours` | Dataset to evaluate on |
| `--use-finetuned` | flag | `False` | Use finetuned model instead of base |
| `--finetune-path` | string | `None` | Path identifier for finetuned model |
| `--device` | string | `cuda:0` | Device for inference |
| `--batch-size` | int | `1` | Batch size (>1 enables batch mode) |
| `--max-samples` | int | `None` | Max samples to evaluate (None = all) |
| `--base-dir` | string | `None` | Custom dataset directory |
| `--log-dir` | string | `eval_all_logs` | Directory for logs |
| `--output-dir` | string | `eval_all_results` | Directory for results |
| `--use-multi-gpu` | flag | `False` | Enable multi-GPU (PAL only) |
| `--gpu-devices` | list | `None` | GPU devices (e.g., `0 1`) |

## Common Usage Examples

### Evaluate Different Models

```bash
# Qwen model
python eval_prompt_match.py --model qwen --dataset ours

# GPT model  
python eval_prompt_match.py --model gpt --dataset ours

# PAL model with multi-GPU
python eval_prompt_match.py --model pal --dataset ours --use-multi-gpu --gpu-devices 0 1
```

### Evaluate on Different Datasets

```bash
# Custom evaluation set
python eval_prompt_match.py --model qwen --dataset ours

# Validation set
python eval_prompt_match.py --model qwen --dataset val

# SynArtifact benchmark
python eval_prompt_match.py --model qwen --dataset synartifact
```

### Batch Processing

```bash
# Single sample mode (default)
python eval_prompt_match.py --model qwen --dataset ours --batch-size 1

# Batch mode for faster inference
python eval_prompt_match.py --model qwen --dataset ours --batch-size 4
```

### Finetuned Models

```bash
# Evaluate finetuned model
python eval_prompt_match.py --model qwen --dataset val \
    --use-finetuned --finetune-path checkpoint-500

# Compare with base model
python eval_prompt_match.py --model qwen --dataset val
```

### Limited Sample Testing

```bash
# Quick test on 10 samples
python eval_prompt_match.py --model qwen --dataset ours --max-samples 10

# Full evaluation
python eval_prompt_match.py --model qwen --dataset ours
```

## Output Structure

### Logs
```
eval_all_logs/
└── {model}/
    └── {finetune_path}/  # if using finetuned
        └── {timestamp}_{dataset}_finetuned.log
```

### Results
```
eval_all_results/
└── {model}/
    └── {finetune_path}/  # if using finetuned
        └── {timestamp}_results_{dataset}.json
```

### JSON Result Format

```json
{
  "0": {
    "process_id": 1,
    "image_path": "path/to/image.png",
    "has_gt_artifacts": true,
    "binary_success": true,
    "classification": "TP",
    "has_pred_artifacts": true,
    "iou": 0.75,
    "loc_precision": 0.85,
    "loc_recall": 0.80,
    "loc_f1": 0.82,
    "legion_miou": 0.73,
    "wsol_iou": 0.71,
    "rouge_l": 0.65,
    "css": 0.72,
    "predictions": {
      "binary": {"prediction": true},
      "localization": [{"bbox_2d": [100, 100, 200, 200]}],
      "explanation": {"explanation": "..."}
    }
  }
}
```

## Evaluation Metrics

### Binary Classification
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Positive class metrics
- **Macro F1**: Average F1 across classes

### Localization
- **Standard**: IoU, Precision, Recall, F1 (bbox-based)
- **LEGION**: Pixel-level segmentation metrics (mIoU, Pixel F1)
- **WSOL**: Threshold-independent IoU

### Explanation
- **ROUGE-L**: Text similarity metric
- **CSS**: Cosine similarity score

## Default Dataset Paths

```python
{
  'synthscars': "/home/jovyan/image-artifacts/data/SynthScars/test",
  'synartifact': "/home/jovyan/image-artifacts/data/SynArtifact/data",
  'loki': "/home/jovyan/image-artifacts/data/loki",
  'richhf': "/home/jovyan/image-artifacts/data/richhf-18k",
  'ours': "/home/jovyan/image-artifacts/data/eval",
  'val': "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/data/artifact_1k.json"
}
```

Override with `--base-dir` if needed.

## Notes

- **Batch Mode**: Use `--batch-size > 1` for faster inference (with OOM fallback)
- **Multi-GPU**: Only supported for PAL model
- **GPT Model**: Tracks API costs automatically
- **Memory**: PAL model clears GPU cache periodically
- **Unified Prompt**: Edit `create_unified_prompt()` function to customize

## Troubleshooting

**OOM Error**: Reduce `--batch-size` or use single-sample mode
**Missing Dataset**: Check path or use `--base-dir` to specify custom location
**Model Not Found**: Ensure model is installed and accessible on specified device

