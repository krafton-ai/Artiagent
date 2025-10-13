"""
Batch Evaluation for Single-Turn VQA Finetuned Models

This evaluation script calculates ALL metrics (binary, localization, explanation)
in a single run without requiring a --type argument.
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
    """Parse JSON response from model output."""
    try:
        # Try to find JSON block in the response
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = response
        
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        logger.warning(f"Raw response (first 500 chars): {response[:500]}")
        logger.warning(f"Raw response (last 200 chars): {response[-200:]}")
        return None


def process_single_vqa_output_all_metrics(raw_output: str, image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Process single-turn VQA model output for ALL evaluation metrics.
    
    Returns dict with predictions for all three evaluation types.
    """
    # Parse JSON response
    parsed = parse_json_response(raw_output)
    
    # Default predictions
    result = {
        'binary': {'prediction': False},
        'localization': [],
        'explanation': {"explanation": "No artifacts detected."},
        'raw_parsed': parsed
    }
    
    if parsed is None:
        # Fallback to text-based detection
        if "yes" in raw_output.lower() and "no" not in raw_output.lower():
            result['binary'] = {'prediction': True}
            result['explanation'] = {"explanation": raw_output}
        return result
    
    # Extract fields from parsed JSON
    artifact_present = parsed.get('artifact_present', 'no').lower() == 'yes'
    bboxes_normalized = parsed.get('bboxes', [])
    explanations = parsed.get('explanations', [])
    caption = parsed.get('caption', '')
    
    # Denormalize bboxes from [0,1] to pixel coordinates
    bboxes_pixel = []
    for bbox in bboxes_normalized:
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x1_pixel = int(x1 * image_width)
            y1_pixel = int(y1 * image_height)
            x2_pixel = int(x2 * image_width)
            y2_pixel = int(y2 * image_height)
            bboxes_pixel.append([x1_pixel, y1_pixel, x2_pixel, y2_pixel])
    
    # Binary prediction
    result['binary'] = {'prediction': artifact_present and len(bboxes_pixel) > 0}
    
    # Localization prediction
    bbox_list = []
    for bbox in bboxes_pixel:
        bbox_list.append({"bbox_2d": bbox})
    result['localization'] = bbox_list
    
    # Explanation prediction
    if caption and explanations:
        explanation_text = f"{caption} " + " ".join(explanations)
    elif caption:
        explanation_text = caption
    elif explanations:
        explanation_text = " ".join(explanations)
    else:
        explanation_text = "No artifacts detected." if not artifact_present else "Artifacts detected."
    result['explanation'] = {"explanation": explanation_text}
    
    return result


class SingleVQAEvaluator:
    """Evaluator for single-turn VQA format models."""
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
    
    def get_prompt(self):
        """Get the single-turn VQA prompt."""
        prompt = """Analyze this image carefully.
Describe whether it contains any visual artifacts,
where those artifacts appear (as bounding boxes normalized to [0,1]),
and provide short explanations for each localized artifact.
Also include a concise caption describing the overall scene.

Return the results as a valid JSON object with the following keys:
- artifact_present: "yes" or "no"
- bboxes: array of [x1, y1, x2, y2] coordinates normalized to [0,1]
- explanations: array of strings describing each artifact
- caption: string describing the overall scene

Generate your response strictly in English only."""
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


def run_single_vqa_evaluation(args):
    """Run comprehensive evaluation for single-turn VQA format."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"single_vqa_comprehensive_{args.dataset}_{exp_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Single-Turn VQA Comprehensive Evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"📊 Evaluating ALL metrics: Binary + Localization + Explanation")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize model
    evaluator_model = SingleVQAEvaluator(args.exp_dir, args.device)
    evaluator_model.load_model()
    
    # Setup dataset
    config = {
        'dataset_type': args.dataset,
        'eval_type': 'localization',  # Placeholder, we evaluate all types
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
    
    # Initialize metrics for all three evaluation types
    all_results = []
    all_binary_success = []
    all_iou_scores = []
    all_loc_f1_scores = []
    all_loc_precision_scores = []
    all_loc_recall_scores = []
    all_rouge_l_scores = []
    all_css_scores = []
    
    # Additional metrics for samples where both GT and prediction have artifacts
    all_iou_scores_both_positive = []
    all_loc_f1_scores_both_positive = []
    all_loc_precision_scores_both_positive = []
    all_loc_recall_scores_both_positive = []
    all_rouge_l_scores_both_positive = []
    all_css_scores_both_positive = []
    
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
                    
                    # Process output for ALL metric types
                    all_predictions = process_single_vqa_output_all_metrics(
                        raw_output, image.size[0], image.size[1]
                    )
                    
                    # Calculate stats for ALL evaluation types
                    binary_stats = evaluator.generate_statistics(
                        args.dataset, 'binary', gt, all_predictions['binary'], image_size=image.size
                    )
                    
                    loc_stats = evaluator.generate_statistics(
                        args.dataset, 'localization', gt, all_predictions['localization'], image_size=image.size
                    )
                    
                    # Additional localization metrics
                    legion_stats = legion_evaluator.generate_statistics(
                        args.dataset, 'localization', gt, all_predictions['localization'], image_size=image.size
                    )
                    wsol_stats = wsol_evaluator.generate_statistics(
                        args.dataset, 'localization', gt, all_predictions['localization'], image_size=image.size
                    )
                    
                    expl_stats = evaluator.generate_statistics(
                        args.dataset, 'explanation', gt, all_predictions['explanation'], image_size=image.size
                    )
                    
                    # Collect binary metrics
                    binary_success = binary_stats.get('binary_success', False)
                    all_binary_success.append(binary_success)
                    
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
                        
                        # Also collect metrics for samples where both GT and prediction have artifacts
                        if binary_success:  # binary_success means prediction also has artifacts
                            all_iou_scores_both_positive.append(iou)
                            all_loc_f1_scores_both_positive.append(loc_f1)
                            all_loc_precision_scores_both_positive.append(loc_precision)
                            all_loc_recall_scores_both_positive.append(loc_recall)
                    
                    # Collect explanation metrics
                    rouge_l = expl_stats.get('rouge_l', 0.0)
                    css = expl_stats.get('css', 0.0)
                    # Ensure no None values
                    rouge_l = rouge_l if rouge_l is not None else 0.0
                    css = css if css is not None else 0.0
                    all_rouge_l_scores.append(rouge_l)
                    all_css_scores.append(css)
                    
                    # Also collect explanation metrics for samples where both GT and prediction have artifacts
                    if has_gt and binary_success:  # Both GT has artifacts AND prediction also has artifacts
                        all_rouge_l_scores_both_positive.append(rouge_l)
                        all_css_scores_both_positive.append(css)
                    
                    # Update progress bar
                    pbar.update(1)
                    iou_val = loc_stats.get('iou', 0.0) if has_gt else 0.0
                    iou_val = iou_val if iou_val is not None else 0.0
                    pbar.set_postfix({
                        'Binary': f'{binary_success}',
                        'IoU': f'{iou_val:.3f}',
                        'ROUGE': f'{rouge_l:.3f}'
                    })
                    
                    # Store comprehensive result
                    result_entry = {
                        'image_path': str(img_path),
                        'ground_truth': gt,
                        'predictions': all_predictions,
                        'raw_output': raw_output,
                        # Binary metrics
                        'binary_success': binary_success,
                        'classification': binary_stats.get('classification'),
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
    logger.info("SINGLE-TURN VQA COMPREHENSIVE EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Binary Classification
    logger.info("\nBINARY CLASSIFICATION:")
    binary_acc = sum(all_binary_success) / len(all_binary_success) if all_binary_success else 0.0
    logger.info(f"  Accuracy: {binary_acc:.4f} ({binary_acc*100:.2f}%)")
    logger.info(f"  Total samples: {len(all_binary_success)}")
    
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
        logger.info(f"  Valid samples (GT positive): {len(all_iou_scores)}")
    else:
        logger.info("  No valid localization samples found")
        mean_iou = mean_f1 = mean_precision = mean_recall = 0.0
    
    # Localization - Both GT and Prediction Positive
    logger.info("\nLOCALIZATION (Both GT & Prediction Positive):")
    if all_iou_scores_both_positive:
        mean_iou_both = sum(all_iou_scores_both_positive) / len(all_iou_scores_both_positive)
        mean_f1_both = sum(all_loc_f1_scores_both_positive) / len(all_loc_f1_scores_both_positive)
        mean_precision_both = sum(all_loc_precision_scores_both_positive) / len(all_loc_precision_scores_both_positive)
        mean_recall_both = sum(all_loc_recall_scores_both_positive) / len(all_loc_recall_scores_both_positive)
        
        logger.info(f"  Mean IoU: {mean_iou_both:.4f}")
        logger.info(f"  Mean F1: {mean_f1_both:.4f}")
        logger.info(f"  Mean Precision: {mean_precision_both:.4f}")
        logger.info(f"  Mean Recall: {mean_recall_both:.4f}")
        logger.info(f"  Valid samples (both positive): {len(all_iou_scores_both_positive)}")
    else:
        logger.info("  No valid localization samples found (both GT and prediction positive)")
        mean_iou_both = mean_f1_both = mean_precision_both = mean_recall_both = 0.0
    
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
    
    # Explanation - Both GT and Prediction Positive
    logger.info("\nEXPLANATION (Both GT & Prediction Positive):")
    if all_rouge_l_scores_both_positive:
        mean_rouge_both = sum(all_rouge_l_scores_both_positive) / len(all_rouge_l_scores_both_positive)
        mean_css_both = sum(all_css_scores_both_positive) / len(all_css_scores_both_positive)
        
        logger.info(f"  Mean ROUGE-L: {mean_rouge_both:.4f}")
        logger.info(f"  Mean CSS: {mean_css_both:.4f}")
        logger.info(f"  Valid samples (both positive): {len(all_rouge_l_scores_both_positive)}")
    else:
        logger.info("  No valid explanation samples found (both GT and prediction positive)")
        mean_rouge_both = mean_css_both = 0.0
    
    logger.info("=" * 80)
    
    # Save results
    results_dir = Path(__file__).parent / "eval_results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"single_vqa_comprehensive_{args.dataset}_{exp_name}_{timestamp}.json"
    
    metrics = {
        'binary': {
            'accuracy': binary_acc,
            'total_samples': len(all_binary_success)
        },
        'localization': {
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'valid_samples': len(all_iou_scores)
        },
        'localization_both_positive': {
            'mean_iou': mean_iou_both,
            'mean_f1': mean_f1_both,
            'mean_precision': mean_precision_both,
            'mean_recall': mean_recall_both,
            'valid_samples': len(all_iou_scores_both_positive)
        },
        'explanation': {
            'mean_rouge_l': mean_rouge,
            'mean_css': mean_css,
            'total_samples': len(all_rouge_l_scores)
        },
        'explanation_both_positive': {
            'mean_rouge_l': mean_rouge_both,
            'mean_css': mean_css_both,
            'valid_samples': len(all_rouge_l_scores_both_positive)
        }
    }
    
    final_results = {
        'config': {
            'exp_dir': args.exp_dir,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'format': 'single_turn_vqa',
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
    parser = argparse.ArgumentParser(description="Single-Turn VQA Comprehensive Evaluation (All Metrics)")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=["ours", "t2i", "synartifact"], help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    run_single_vqa_evaluation(args)


if __name__ == "__main__":
    main()

