"""
Batch Evaluation for Multi-Turn VQA Finetuned Models

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


def process_multi_vqa_output_all_metrics(raw_outputs: List[str], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Process multi-turn VQA model outputs for ALL evaluation metrics.
    
    Args:
        raw_outputs: List of responses from the multi-turn conversation
        image_width: Image width for bbox denormalization
        image_height: Image height for bbox denormalization
    
    Returns:
        Dict with predictions for all three evaluation types
    """
    # Default predictions
    result = {
        'binary': {'prediction': False},
        'localization': [],
        'explanation': {"explanation": "No artifacts detected."},
        'raw_parsed': raw_outputs
    }
    
    if not raw_outputs or len(raw_outputs) < 2:
        return result
    
    # Q1 response: "yes" or "no"
    q1_response = raw_outputs[0].strip().lower()
    has_artifacts = "yes" in q1_response and "no" not in q1_response
    
    if not has_artifacts:
        # Negative sample - no artifacts
        result['binary'] = {'prediction': False}
        if len(raw_outputs) >= 4:
            # Q4 response for negative samples
            result['explanation'] = {"explanation": raw_outputs[-1]}
        return result
    
    # Positive sample - has artifacts
    result['binary'] = {'prediction': True}
    
    # Q2 response: JSON array of bboxes
    if len(raw_outputs) >= 2:
        q2_response = raw_outputs[1]
        try:
            # Try to parse bboxes from Q2 response
            bboxes_normalized = json.loads(q2_response)
            if isinstance(bboxes_normalized, list):
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
                
                # Localization prediction
                bbox_list = []
                for bbox in bboxes_pixel:
                    bbox_list.append({"bbox_2d": bbox})
                result['localization'] = bbox_list
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse Q2 bbox response: {e}")
            logger.warning(f"Q2 response: {q2_response[:200]}")
    
    # Collect explanations from Q3 responses and caption from Q4
    explanations = []
    caption = ""
    
    # Q3 responses: explanations for each bbox (every other response starting from index 2)
    for i in range(2, len(raw_outputs) - 1, 2):
        if i < len(raw_outputs):
            explanations.append(raw_outputs[i])
    
    # Q4 response: caption (last response)
    if raw_outputs:
        caption = raw_outputs[-1]
    
    # Combine explanations and caption
    if caption and explanations:
        explanation_text = f"{caption} " + " ".join(explanations)
    elif caption:
        explanation_text = caption
    elif explanations:
        explanation_text = " ".join(explanations)
    else:
        explanation_text = "Artifacts detected."
    
    result['explanation'] = {"explanation": explanation_text}
    
    return result


class MultiVQAEvaluator:
    """Evaluator for multi-turn VQA format models."""
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
    
    def get_prompts(self):
        """Get the multi-turn VQA prompts."""
        prompts = {
            'Q1': "Does this image contain any visual artifacts?",
            'Q2': "Please provide the bounding boxes of the artifacts as a JSON array of [x1, y1, x2, y2] coordinates normalized to [0,1].",
            'Q3_template': "What is this artifact? Please describe the artifact at bbox {bbox}.",
            'Q4_pos': "Please provide a concise caption describing the overall scene.",
            'Q4_neg': "Please provide a concise caption describing the overall scene."
        }
        return prompts
    
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
    
    def inference_single_turn(self, image: Image.Image, prompt: str) -> str:
        """Perform single-turn inference."""
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
        
        # Get multimodal inputs
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            images=[image], videos=[], audios=[],
            imglens=[1], vidlens=[0], audlens=[0],
            batch_ids=[prompt_ids], processor=self.processor
        )
        
        # Prepare batch
        batch = self.tokenizer.pad(
            {"input_ids": [prompt_ids]},
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
        
        input_len = batch["attention_mask"][0].sum().item()
        response_ids = outputs[0, input_len:]
        response = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        return response
    
    def inference_multi_turn(self, image: Image.Image, prompts: Dict[str, str], has_artifacts: bool = True) -> List[str]:
        """Perform multi-turn inference following the conversation flow."""
        responses = []
        
        # Q1: Does this image contain any visual artifacts?
        q1_response = self.inference_single_turn(image, prompts['Q1'])
        responses.append(q1_response)
        
        # Check if artifacts are present
        q1_lower = q1_response.strip().lower()
        has_artifacts = "yes" in q1_lower and "no" not in q1_lower
        
        if has_artifacts:
            # Q2: Get bounding boxes
            q2_response = self.inference_single_turn(image, prompts['Q2'])
            responses.append(q2_response)
            
            # Parse bboxes for Q3 questions
            try:
                bboxes = json.loads(q2_response)
                if isinstance(bboxes, list):
                    # Q3: Ask about each bbox
                    for bbox in bboxes:
                        bbox_str = json.dumps(bbox)
                        q3_prompt = prompts['Q3_template'].format(bbox=bbox_str)
                        q3_response = self.inference_single_turn(image, q3_prompt)
                        responses.append(q3_response)
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Failed to parse Q2 bbox response for Q3 questions")
            
            # Q4: Get caption
            q4_response = self.inference_single_turn(image, prompts['Q4_pos'])
            responses.append(q4_response)
        else:
            # Q4: Get caption for negative sample
            q4_response = self.inference_single_turn(image, prompts['Q4_neg'])
            responses.append(q4_response)
        
        return responses


def run_multi_vqa_evaluation(args):
    """Run comprehensive evaluation for multi-turn VQA format."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"multi_vqa_comprehensive_{args.dataset}_{exp_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Multi-Turn VQA Comprehensive Evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"📊 Evaluating ALL metrics: Binary + Localization + Explanation")
    
    # Initialize model
    evaluator_model = MultiVQAEvaluator(args.exp_dir, args.device)
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
    
    # Get prompts
    prompts = evaluator_model.get_prompts()
    
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
    
    total_processed = 0
    total_samples = args.max_samples if args.max_samples else len(data_iterator)
    pbar = tqdm(total=total_samples, desc="Evaluating samples", unit="sample")
    
    for gt, image_path in data_iterator:
        if args.max_samples and total_processed >= args.max_samples:
            break
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Determine if sample has artifacts for conversation flow
            has_gt_artifacts = gt.get('has_artifacts', True) if args.dataset == 'ours' else bool(gt.get('Artifacts annotation', []))
            
            # Perform multi-turn inference
            raw_outputs = evaluator_model.inference_multi_turn(image, prompts, has_gt_artifacts)
            
            # Process output for ALL metric types
            all_predictions = process_multi_vqa_output_all_metrics(
                raw_outputs, image.size[0], image.size[1]
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
            if has_gt_artifacts:
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
            if has_gt_artifacts and binary_success:  # Both GT has artifacts AND prediction also has artifacts
                all_rouge_l_scores_both_positive.append(rouge_l)
                all_css_scores_both_positive.append(css)
            
            # Update progress bar
            pbar.update(1)
            iou_val = loc_stats.get('iou', 0.0) if has_gt_artifacts else 0.0
            iou_val = iou_val if iou_val is not None else 0.0
            pbar.set_postfix({
                'Binary': f'{binary_success}',
                'IoU': f'{iou_val:.3f}',
                'ROUGE': f'{rouge_l:.3f}'
            })
            
            # Store comprehensive result
            result_entry = {
                'image_path': str(image_path),
                'ground_truth': gt,
                'predictions': all_predictions,
                'raw_outputs': raw_outputs,
                # Binary metrics
                'binary_success': binary_success,
                'classification': binary_stats.get('classification'),
                # Localization metrics
                'iou': loc_stats.get('iou') if has_gt_artifacts else None,
                'loc_f1': loc_stats.get('loc_f1') if has_gt_artifacts else None,
                'loc_precision': loc_stats.get('loc_precision') if has_gt_artifacts else None,
                'loc_recall': loc_stats.get('loc_recall') if has_gt_artifacts else None,
                'legion_iou': legion_stats.get('iou') if has_gt_artifacts else None,
                'wsol_iou': wsol_stats.get('iou') if has_gt_artifacts else None,
                # Explanation metrics
                'rouge_l': rouge_l,
                'css': css,
                'has_gt_artifacts': has_gt_artifacts
            }
            
            all_results.append(result_entry)
            total_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {image_path}: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(1)
            total_processed += 1
    
    pbar.close()
    
    # Calculate final metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("MULTI-TURN VQA COMPREHENSIVE EVALUATION RESULTS")
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
    results_file = results_dir / f"multi_vqa_comprehensive_{args.dataset}_{exp_name}_{timestamp}.json"
    
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
            'max_samples': args.max_samples,
            'format': 'multi_turn_vqa',
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
    parser = argparse.ArgumentParser(description="Multi-Turn VQA Comprehensive Evaluation (All Metrics)")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=["ours", "t2i", "synartifact"], help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    run_multi_vqa_evaluation(args)


if __name__ == "__main__":
    main()
