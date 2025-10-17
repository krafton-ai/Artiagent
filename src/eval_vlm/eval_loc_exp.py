"""
Localization + Explanation Evaluation for Finetuned VLM

This evaluation script focuses only on localization and explanation metrics,
removing binary classification to streamline the evaluation process.
"""

import argparse
import json
import logging
import os
import re
import sys
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
    process_finetuned_output,
    setup_logging,
)

# Import from parent eval directory
sys.path.append(str(Path(__file__).parent.parent / "eval"))
from eval_utils import Evaluation
import legion_eval_utils
import wsol_eval_utils

logger = logging.getLogger(__name__)


class LocExpEvaluator:
    """Evaluator for localization and explanation only."""
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
    
    def get_prompt(self):
        """Get the localization + explanation prompt."""
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
        
        self.model = load_model(self.tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
        # Don't move model to device if it's already dispatched with accelerate
        if not hasattr(self.model, 'hf_device_map') or self.model.hf_device_map is None:
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


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from model output with two fenced JSON blocks."""
    try:
        # Find all fenced JSON blocks
        json_pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if len(matches) >= 2:
            # Parse first block (artifacts array)
            artifacts_json = matches[0]
            artifacts_data = json.loads(artifacts_json)
            
            # Parse second block (explanation object)
            explanation_json = matches[1]
            explanation_data = json.loads(explanation_json)
            
            # Combine into expected format
            result = {
                'artifacts': artifacts_data,
                'explanation': explanation_data.get('explanation', '')
            }
            return result
            
        elif len(matches) == 1:
            # Try to parse single block
            json_str = matches[0]
            parsed = json.loads(json_str)
            
            # Check if it's the old format or new format
            if isinstance(parsed, dict) and 'bboxes' in parsed:
                # Old format - convert to new format
                artifacts = []
                for i, bbox in enumerate(parsed.get('bboxes', [])):
                    label = parsed.get('explanations', [''])[i] if i < len(parsed.get('explanations', [])) else ''
                    artifacts.append({
                        'bbox_2d': bbox,
                        'label': label
                    })
                result = {
                    'artifacts': artifacts,
                    'explanation': parsed.get('caption', '')
                }
                return result
            elif isinstance(parsed, list):
                # New format - artifacts array only
                result = {
                    'artifacts': parsed,
                    'explanation': ''
                }
                return result
            else:
                # Single explanation object
                result = {
                    'artifacts': [],
                    'explanation': parsed.get('explanation', '')
                }
                return result
        else:
            # Try to find raw JSON without fences
            json_pattern = r'(\[.*?\]|\{.*?\})'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Try to parse the first match
                json_str = matches[0]
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    result = {
                        'artifacts': parsed,
                        'explanation': ''
                    }
                    return result
                elif isinstance(parsed, dict):
                    if 'bboxes' in parsed:
                        # Old format
                        artifacts = []
                        for i, bbox in enumerate(parsed.get('bboxes', [])):
                            label = parsed.get('explanations', [''])[i] if i < len(parsed.get('explanations', [])) else ''
                            artifacts.append({
                                'bbox_2d': bbox,
                                'label': label
                            })
                        result = {
                            'artifacts': artifacts,
                            'explanation': parsed.get('caption', '')
                        }
                        return result
                    else:
                        # Single explanation
                        result = {
                            'artifacts': [],
                            'explanation': parsed.get('explanation', '')
                        }
                        return result
        
        # If nothing found, return empty result
        return {
            'artifacts': [],
            'explanation': ''
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        logger.warning(f"Raw response (first 500 chars): {response[:500]}")
        logger.warning(f"Raw response (last 200 chars): {response[-200:]}")
        return {
            'artifacts': [],
            'explanation': ''
        }


def process_loc_exp_output(raw_output: str, image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Process model output for localization and explanation evaluation only.
    
    Returns dict with predictions for localization and explanation.
    """
    # Try to parse as JSON first
    parsed_json = parse_json_response(raw_output)
    
    if parsed_json is not None:
        # Extract data from new JSON structure
        artifacts = parsed_json.get('artifacts', [])
        explanation_text = parsed_json.get('explanation', '')
        
        # Format bboxes for localization evaluation
        bbox_list = []
        for artifact in artifacts:
            if isinstance(artifact, dict) and 'bbox_2d' in artifact:
                bbox = artifact['bbox_2d']
                if len(bbox) == 4:
                    try:
                        # Coordinates are already in pixel format
                        x1, y1, x2, y2 = bbox
                        # Ensure coordinates are valid numbers
                        if all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                            # Validate coordinates are within image bounds
                            x1 = max(0, min(image_width, int(x1)))
                            y1 = max(0, min(image_height, int(y1)))
                            x2 = max(0, min(image_width, int(x2)))
                            y2 = max(0, min(image_height, int(y2)))
                            
                            # Ensure x2 > x1 and y2 > y1
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
    else:
        # Fallback to old processing method
        logger.warning("JSON parsing failed, falling back to regex-based parsing")
        localization_prediction = process_finetuned_output(raw_output, 'localization')
        explanation_prediction = process_finetuned_output(raw_output, 'explanation')
        
        return {
            'localization': localization_prediction,
            'explanation': explanation_prediction,
            'raw_output': raw_output
        }


def run_loc_exp_evaluation(args):
    """Run evaluation for localization and explanation only."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"loc_exp_{args.dataset}_{exp_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🎯 Starting Localization + Explanation Evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"📊 Evaluating: Localization + Explanation")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize model
    evaluator_model = LocExpEvaluator(args.exp_dir, args.device)
    evaluator_model.load_model()
    
    # Setup dataset
    config = {
        'dataset_type': args.dataset,
        'eval_type': 'localization',  # We evaluate both localization and explanation
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
    
    # Initialize metrics for localization and explanation
    all_results = []
    all_iou_scores = []
    all_loc_f1_scores = []
    all_loc_precision_scores = []
    all_loc_recall_scores = []
    all_rouge_l_scores = []
    all_css_scores = []
    json_parse_success_count = 0
    total_samples_count = 0
    
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
                    
                    # Process output for localization and explanation only
                    predictions = process_loc_exp_output(
                        raw_output, image.size[0], image.size[1]
                    )
                    
                    # Log JSON parsing success/failure for debugging
                    total_samples_count += 1
                    if 'parsed_json' in predictions:
                        json_parse_success_count += 1
                        logger.debug(f"Successfully parsed JSON for {img_path}")
                    else:
                        logger.warning(f"JSON parsing failed for {img_path}, using fallback method")
                    
                    # Calculate stats for localization and explanation
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
                    
                    # Collect localization metrics (only for positive samples)
                    has_gt = gt.get('has_artifacts', True) if args.dataset == 'ours' else bool(gt.get('Artifacts annotation', []))
                    
                    if has_gt:
                        iou = loc_stats.get('iou', 0.0)
                        loc_f1 = loc_stats.get('loc_f1', 0.0)
                        loc_precision = loc_stats.get('loc_precision', 0.0)
                        loc_recall = loc_stats.get('loc_recall', 0.0)
                        
                        # Ensure no None values
                        iou = iou if iou is not None else 0.0
                        loc_f1 = loc_f1 if loc_f1 is not None else 0.0
                        loc_precision = loc_precision if loc_precision is not None else 0.0
                        loc_recall = loc_recall if loc_recall is not None else 0.0
                        
                        all_iou_scores.append(iou)
                        all_loc_f1_scores.append(loc_f1)
                        all_loc_precision_scores.append(loc_precision)
                        all_loc_recall_scores.append(loc_recall)
                    
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
    logger.info("LOCALIZATION + EXPLANATION EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Localization
    logger.info("\nLOCALIZATION:")
    if all_iou_scores:
        mean_iou = sum(all_iou_scores) / len(all_iou_scores)
        mean_f1 = sum(all_loc_f1_scores) / len(all_loc_f1_scores)
        mean_precision = sum(all_loc_precision_scores) / len(all_loc_precision_scores)
        mean_recall = sum(all_loc_recall_scores) / len(all_loc_recall_scores)
        
        logger.info(f"  Mean IoU: {mean_iou:.4f}")
        logger.info(f"  Mean F1: {mean_f1:.4f}")
        logger.info(f"  Mean Precision: {mean_precision:.4f}")
        logger.info(f"  Mean Recall: {mean_recall:.4f}")
        logger.info(f"  Valid samples (positive): {len(all_iou_scores)}")
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
        
        # Valid samples only (positive)
        valid_results = [r for r in all_results if r.get('has_gt_artifacts')]
        if valid_results:
            valid_rouge = sum(r['rouge_l'] for r in valid_results) / len(valid_results)
            valid_css = sum(r['css'] for r in valid_results) / len(valid_results)
            logger.info(f"  Valid (positive) samples: {len(valid_results)}")
            logger.info(f"    Mean ROUGE-L: {valid_rouge:.4f}")
            logger.info(f"    Mean CSS: {valid_css:.4f}")
    else:
        logger.info("  No explanation samples found")
        mean_rouge = mean_css = 0.0
    
    # JSON Parsing Statistics
    logger.info("\nJSON PARSING:")
    if total_samples_count > 0:
        json_success_rate = (json_parse_success_count / total_samples_count) * 100
        logger.info(f"  JSON parsing success rate: {json_success_rate:.1f}% ({json_parse_success_count}/{total_samples_count})")
    else:
        logger.info("  No samples processed")
        json_success_rate = 0.0
    
    logger.info("=" * 80)
    
    # Save results
    results_dir = Path(__file__).parent / "eval_results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"loc_exp_{args.dataset}_{exp_name}_{timestamp}.json"
    
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
        },
        'json_parsing': {
            'success_rate': json_success_rate,
            'successful_parses': json_parse_success_count,
            'total_samples': total_samples_count
        }
    }
    
    final_results = {
        'config': {
            'exp_dir': args.exp_dir,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'format': 'loc_exp',
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
    parser = argparse.ArgumentParser(description="Localization + Explanation Evaluation")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=['synthscars', 'synartifact', 'loki', 'richhf', 'ours', 'val'], help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    run_loc_exp_evaluation(args)


if __name__ == "__main__":
    main()
