"""
Batch Evaluation for Single-Turn Localization + Explanation Finetuned Models

This evaluation script calculates localization and explanation metrics
for single-turn VQA models that output two fenced JSON blocks.
"""

import argparse
import json
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "train" / "LLaMA-Factory" / "src"))

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import ModelArguments, DataArguments, EvaluationArguments, get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data.mm_plugin import IMAGE_PLACEHOLDER

from eval_batch_utils import (
    DatasetIterator,
    setup_logging,
)

# Import from parent eval directory
sys.path.append(str(Path(__file__).parent.parent / "eval"))
from eval_utils import Evaluation
import legion_eval_utils
import wsol_eval_utils

logger = logging.getLogger(__name__)


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse two fenced JSON blocks from model output."""
    try:
        # Find multiple fenced JSON blocks
        json_pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if len(matches) >= 2:
            # First block: artifacts array
            artifacts_json = matches[0]
            artifacts_data = json.loads(artifacts_json)
            
            # Second block: explanation object
            explanation_json = matches[1]
            explanation_data = json.loads(explanation_json)
            
            result = {
                'artifacts': artifacts_data,
                'explanation': explanation_data.get('explanation', '')
            }
            return result
        
        elif len(matches) == 1:
            # Single fenced block - try to parse as artifacts array
            try:
                artifacts_data = json.loads(matches[0])
                if isinstance(artifacts_data, list):
                    return {
                        'artifacts': artifacts_data,
                        'explanation': 'Artifacts detected.'
                    }
                else:
                    return {
                        'artifacts': [],
                        'explanation': explanation_data.get('explanation', 'No artifacts detected.')
                    }
            except:
                return {
                    'artifacts': [],
                    'explanation': 'Failed to parse response.'
                }
        
        else:
            # No fenced blocks - try to find raw JSON
            json_pattern = r'(\[.*?\]|\{.*?\})'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                try:
                    parsed = json.loads(matches[0])
                    if isinstance(parsed, list):
                        return {
                            'artifacts': parsed,
                            'explanation': 'Artifacts detected.'
                        }
                    else:
                        return {
                            'artifacts': [],
                            'explanation': parsed.get('explanation', 'No artifacts detected.')
                        }
                except:
                    pass
            
            # Fallback: return empty
            return {
                'artifacts': [],
                'explanation': 'No artifacts detected.'
            }
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return {
            'artifacts': [],
            'explanation': 'Failed to parse response.'
        }


def process_single_turn_loc_exp_output(raw_output: str, image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Process single-turn localization + explanation model output.
    
    Returns dict with localization and explanation predictions.
    """
    parsed_json = parse_json_response(raw_output)
    
    if parsed_json is not None:
        artifacts = parsed_json.get('artifacts', [])
        explanation_text = parsed_json.get('explanation', '')
        
        # Process bboxes (already in pixel coordinates)
        bbox_list = []
        for artifact in artifacts:
            if isinstance(artifact, dict) and 'bbox_2d' in artifact:
                bbox = artifact['bbox_2d']
                if len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = bbox
                        if all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                            # Validate and clamp coordinates
                            x1 = max(0, min(image_width, int(x1)))
                            y1 = max(0, min(image_height, int(y1)))
                            x2 = max(0, min(image_width, int(x2)))
                            y2 = max(0, min(image_height, int(y2)))
                            
                            # Ensure valid bbox (x2 > x1, y2 > y1)
                            if x2 > x1 and y2 > y1:
                                bbox_list.append({"bbox_2d": [x1, y1, x2, y2]})
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid bbox coordinates: {bbox}, error: {e}")
                        continue
        
        return {
            'localization': bbox_list,
            'explanation': {"explanation": explanation_text},
            'raw_output': raw_output,
            'parsed_json': parsed_json
        }
    
    # Fallback
    return {
        'localization': [],
        'explanation': {"explanation": "No artifacts detected."},
        'raw_output': raw_output,
        'parsed_json': None
    }


class SingleTurnLocExpEvaluator:
    """Evaluator for single-turn localization + explanation format models."""
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
    
    def get_prompt(self):
        """Get the single-turn localization + explanation prompt."""
        prompt = """Analyze this image carefully and identify any visual artifacts present.

You must respond with exactly two fenced JSON blocks in this order:

First JSON block - Array of artifacts with pixel coordinates:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "description of the artifact in this region"},
  {"bbox_2d": [x1, y1, x2, y2], "label": "description of the artifact in this region"}
]
```

Second JSON block - Explanation:
```json
{"explanation": "description of the anomalies in this image."}
```

Requirements:
- Use pixel coordinates (not normalized)
- Each bbox_2d array must have exactly 4 numbers: [x_min, y_min, x_max, y_max]
- Provide explanations in English only
- Ensure both JSON blocks are properly formatted and valid."""
        return prompt
    
    def load_model(self):
        """Load model using LLaMA-Factory's evaluation framework."""
        logger.info("Loading model using LLaMA-Factory evaluation framework...")
        
        args_dict = {
            "model_name_or_path": self.exp_dir,
            "template": "qwen2_vl",
            "task": "mmlu",
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
        }
        
        args_list = []
        for key, value in args_dict.items():
            args_list.append(f"--{key}")
            args_list.append(str(value))
        
        model_args, data_args, eval_args, finetuning_args = get_eval_args(args_list)
        
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module.get("processor")
        
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        
        self.model = load_model(self.tokenizer, model_args, finetuning_args)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """Perform batch inference."""
        batch_size = len(images)
        
        # Process each sample
        batch_input_ids = []
        for image in images:
            messages = [{"role": "user", "content": f"{IMAGE_PLACEHOLDER}{prompt}"}]
            
            if hasattr(self.template, 'mm_plugin') and self.template.mm_plugin:
                processed_messages = self.template.mm_plugin.process_messages(
                    messages, [image], [], [], self.processor
                )
            else:
                processed_messages = messages
            
            paired_messages = processed_messages + [{"role": "assistant", "content": ""}]
            prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages)
            
            if hasattr(self.template, 'mm_plugin') and self.template.mm_plugin:
                prompt_ids, _ = self.template.mm_plugin.process_token_ids(
                    prompt_ids, None, [image], [], [], self.tokenizer, self.processor
                )
            
            batch_input_ids.append(prompt_ids)
        
        # Get multimodal inputs
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            images=images, videos=[], audios=[],
            imglens=[1] * batch_size, vidlens=[0] * batch_size, audlens=[0] * batch_size,
            batch_ids=batch_input_ids, processor=self.processor
        )
        
        # Pad and prepare batch
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = {
            **batch,
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        for key, value in mm_inputs.items():
            if hasattr(value, 'to'):
                gen_kwargs[key] = value.to(self.model.device)
            else:
                gen_kwargs[key] = value
        
        with torch.inference_mode():
            # Disable tqdm progress bars during generation
            import os
            old_tqdm_disable = os.environ.get('TQDM_DISABLE', None)
            os.environ['TQDM_DISABLE'] = '1'
            try:
                outputs = self.model.generate(**gen_kwargs)
            finally:
                if old_tqdm_disable is not None:
                    os.environ['TQDM_DISABLE'] = old_tqdm_disable
                else:
                    os.environ.pop('TQDM_DISABLE', None)
        
        results = []
        for i in range(batch_size):
            input_len = batch["attention_mask"][i].sum().item()
            response_ids = outputs[i, input_len:]
            response = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            results.append(response)
        
        return results


def run_single_turn_loc_exp_evaluation(args):
    """Run evaluation for single-turn localization + explanation format."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"single_turn_loc_exp_{args.dataset}_{exp_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Single-Turn Localization + Explanation Evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize model
    evaluator_model = SingleTurnLocExpEvaluator(args.exp_dir, args.device)
    evaluator_model.load_model()
    
    # Setup dataset
    config = {
        'dataset_type': args.dataset,
        'eval_type': 'localization',
        'data_path': args.dataset_path,
        'base_dir': args.dataset_path,
    }
    data_iterator = DatasetIterator(config)
    
    # Setup evaluation metrics
    evaluator = Evaluation()
    legion_evaluator = legion_eval_utils.Evaluation()
    wsol_evaluator = wsol_eval_utils.Evaluation()
    
    # Get prompt
    input_query = evaluator_model.get_prompt()
    
    # Initialize metrics
    all_results = []
    all_iou_scores = []
    all_pixel_f1_scores = []
    all_pixel_precision_scores = []
    all_pixel_recall_scores = []
    all_rouge_l_scores = []
    all_css_scores = []
    
    batch_images = []
    batch_metadata = []
    total_processed = 0
    
    total_samples = args.max_samples if args.max_samples else len(data_iterator)
    pbar = tqdm(total=total_samples, desc="Evaluating samples", unit="sample")
    
    for gt, image_path in data_iterator:
        if args.max_samples and total_processed >= args.max_samples:
            break
        
        image = Image.open(image_path).convert('RGB')
        batch_images.append(image)
        batch_metadata.append((image_path, gt))
        
        if len(batch_images) == args.batch_size or total_processed + len(batch_images) >= total_samples:
            try:
                batch_raw_outputs = evaluator_model.inference_batch(batch_images, input_query)
                
                for i, (raw_output, (img_path, gt)) in enumerate(zip(batch_raw_outputs, batch_metadata)):
                    image = batch_images[i]
                    
                    # Process output
                    predictions = process_single_turn_loc_exp_output(
                        raw_output, image.size[0], image.size[1]
                    )
                    
                    # Calculate stats
                    loc_stats = evaluator.generate_statistics(
                        args.dataset, 'localization', gt, predictions['localization'], image_size=image.size
                    )
                    
                    # Additional localization metrics
                    legion_stats = legion_evaluator.generate_statistics(
                        args.dataset, 'localization', gt, predictions['localization'], image_size=image.size
                    )
                    wsol_stats = wsol_evaluator.generate_statistics(
                        args.dataset, 'localization', gt, predictions['localization'], image_size=image.size
                    )
                    
                    expl_stats = evaluator.generate_statistics(
                        args.dataset, 'explanation', gt, predictions['explanation'], image_size=image.size
                    )
                    
                    # Collect metrics
                    if args.dataset in ['synthscars', 'loki', 'richhf']:
                        # These datasets always have artifacts (they are artifact detection datasets)
                        has_gt = True
                    elif args.dataset in ['ours', 'val']:
                        # Check has_artifacts field
                        has_gt = gt.get('has_artifacts', True)
                    elif args.dataset == 'synartifact':
                        # Check Artifacts annotation field
                        has_gt = bool(gt.get('Artifacts annotation', []))
                    else:
                        # Default fallback
                        has_gt = True
                    
                    if has_gt:
                        iou = legion_stats.get('iou', 0.0)
                        pixel_f1 = legion_stats.get('pixel_f1', 0.0)
                        pixel_precision = legion_stats.get('pixel_precision', 0.0)
                        pixel_recall = legion_stats.get('pixel_recall', 0.0)
                        
                        # Ensure no None values
                        iou = iou if iou is not None else 0.0
                        pixel_f1 = pixel_f1 if pixel_f1 is not None else 0.0
                        pixel_precision = pixel_precision if pixel_precision is not None else 0.0
                        pixel_recall = pixel_recall if pixel_recall is not None else 0.0
                        
                        all_iou_scores.append(iou)
                        all_pixel_f1_scores.append(pixel_f1)
                        all_pixel_precision_scores.append(pixel_precision)
                        all_pixel_recall_scores.append(pixel_recall)
                    
                    # Collect explanation metrics
                    rouge_l = expl_stats.get('rouge_l', 0.0)
                    css = expl_stats.get('css', 0.0)
                    # Ensure no None values
                    rouge_l = rouge_l if rouge_l is not None else 0.0
                    css = css if css is not None else 0.0
                    all_rouge_l_scores.append(rouge_l)
                    all_css_scores.append(css)
                    
                    # Update progress bar
                    pbar.update(1)
                    iou_val = loc_stats.get('iou', 0.0) if has_gt else 0.0
                    iou_val = iou_val if iou_val is not None else 0.0
                    pbar.set_postfix({
                        'IoU': f'{iou_val:.3f}',
                        'ROUGE': f'{rouge_l:.3f}'
                    })
                    
                    # Store result
                    result_entry = {
                        'image_path': str(img_path),
                        'ground_truth': gt,
                        'predictions': predictions,
                        'raw_output': raw_output,
                        # Localization metrics
                        'iou': loc_stats.get('iou') if has_gt else None,
                        'loc_f1': loc_stats.get('loc_f1') if has_gt else None,
                        'loc_precision': loc_stats.get('loc_precision') if has_gt else None,
                        'loc_recall': loc_stats.get('loc_recall') if has_gt else None,
                        'legion_iou': legion_stats.get('iou') if has_gt else None,
                        'legion_pixel_f1': legion_stats.get('pixel_f1') if has_gt else None,
                        'wsol_iou': wsol_stats.get('iou') if has_gt else None,
                        # Explanation metrics
                        'rouge_l': rouge_l,
                        'css': css,
                        'has_gt_artifacts': has_gt
                    }
                    
                    all_results.append(result_entry)
                    total_processed += 1
                    
            except Exception as e:
                logger.error(f"Error during batch evaluation: {e}")
                import traceback
                traceback.print_exc()
            
            batch_images = []
            batch_metadata = []
    
    pbar.close()
    
    # Calculate final metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("SINGLE-TURN LOCALIZATION + EXPLANATION EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Localization
    logger.info("\nLOCALIZATION:")
    if all_iou_scores:
        mean_iou = sum(all_iou_scores) / len(all_iou_scores)
        mean_f1 = sum(all_pixel_f1_scores) / len(all_pixel_f1_scores)
        mean_precision = sum(all_pixel_precision_scores) / len(all_pixel_precision_scores)
        mean_recall = sum(all_pixel_recall_scores) / len(all_pixel_recall_scores)
        
        logger.info(f"  Mean IoU: {mean_iou:.4f}")
        logger.info(f"  Mean F1: {mean_f1:.4f}")
        logger.info(f"  Mean Precision: {mean_precision:.4f}")
        logger.info(f"  Mean Recall: {mean_recall:.4f}")
        logger.info(f"  Valid samples (GT positive): {len(all_iou_scores)}")
    else:
        logger.info("  No valid localization samples found")
        mean_iou = mean_f1 = mean_precision = mean_recall = 0.0
    
    # Explanation
    logger.info("\nEXPLANATION:")
    if all_rouge_l_scores:
        mean_rouge = sum(all_rouge_l_scores) / len(all_rouge_l_scores)
        mean_css = sum(all_css_scores) / len(all_css_scores)
        
        logger.info(f"  Mean ROUGE-L: {mean_rouge:.4f}")
        logger.info(f"  Mean CSS: {mean_css:.4f}")
        logger.info(f"  Total samples: {len(all_rouge_l_scores)}")
    else:
        logger.info("  No explanation samples found")
        mean_rouge = mean_css = 0.0
    
    logger.info("=" * 80)
    
    # Save results
    results_dir = Path(__file__).parent / "eval_results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"single_turn_loc_exp_{args.dataset}_{exp_name}_{timestamp}.json"
    
    metrics = {
        'localization': {
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'valid_samples': len(all_iou_scores)
        },
        'explanation': {
            'mean_rouge_l': mean_rouge,
            'mean_css': mean_css,
            'total_samples': len(all_rouge_l_scores)
        }
    }
    
    final_results = {
        'config': {
            'exp_dir': args.exp_dir,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'format': 'single_turn_loc_exp',
            'timestamp': timestamp
        },
        'metrics': metrics,
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    logger.info(f"✅ Log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Single-Turn Localization + Explanation Evaluation")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=["ours", "synthscars", "synartifact", "loki", "richhf", "val"], help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    run_single_turn_loc_exp_evaluation(args)


if __name__ == "__main__":
    main()