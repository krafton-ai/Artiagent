"""
Batch Evaluation for Multi-Turn Localization + Explanation Finetuned Models

This evaluation script calculates localization and explanation metrics
for multi-turn VQA models that follow the progressive Q&A format.
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


def process_multi_turn_loc_exp_output(raw_outputs: List[str], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Process multi-turn localization + explanation model outputs.
    
    Args:
        raw_outputs: List of responses from the multi-turn conversation
        image_width: Image width for bbox validation
        image_height: Image height for bbox validation
    
    Returns:
        Dict with localization and explanation predictions
    """
    # Default predictions
    result = {
        'localization': [],
        'explanation': {"explanation": "No artifacts detected."},
        'raw_outputs': raw_outputs
    }
    
    if not raw_outputs or len(raw_outputs) < 2:
        return result
    
    # Q1 response: bbox coordinates (fenced JSON array)
    if len(raw_outputs) >= 1:
        q1_response = raw_outputs[0]
        try:
            # Try to parse bboxes from Q1 response (fenced JSON)
            json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
            match = re.search(json_pattern, q1_response, re.DOTALL)
            
            if match:
                bboxes_json = match.group(1)
                bboxes = json.loads(bboxes_json)
            else:
                # Try raw JSON
                bboxes = json.loads(q1_response)
            
            if isinstance(bboxes, list):
                # Process bboxes (already in pixel coordinates)
                bbox_list = []
                for bbox in bboxes:
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
                
                result['localization'] = bbox_list
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse Q1 bbox response: {e}")
            logger.warning(f"Q1 response: {q1_response[:200]}")
    
    # Collect explanations from Q2 responses and caption from Q3
    explanations = []
    caption = ""
    
    # Q2 responses: explanations for each bbox (every other response starting from index 1)
    for i in range(1, len(raw_outputs) - 1, 1):
        if i < len(raw_outputs):
            explanations.append(raw_outputs[i])
    
    # Q3 response: caption (last response, fenced JSON)
    if raw_outputs:
        q3_response = raw_outputs[-1]
        try:
            # Try to parse caption from Q3 response (fenced JSON)
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, q3_response, re.DOTALL)
            
            if match:
                caption_json = match.group(1)
                caption_data = json.loads(caption_json)
                caption = caption_data.get('explanation', q3_response)
            else:
                caption = q3_response
        except (json.JSONDecodeError, ValueError, TypeError):
            caption = q3_response
    
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


class MultiTurnLocExpEvaluator:
    """Evaluator for multi-turn localization + explanation format models."""
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
    
    def get_prompts(self):
        """Get the multi-turn localization + explanation prompts."""
        prompts = {
            'Q1': "Examine the image carefully and identify any visual artifacts. List the artifact regions as bounding boxes in coordinates [x1, y1, x2, y2].",
            'Q2_template': "For the region at {bbox_str}, briefly describe what is wrong there. Return a short sentence only.",
            'Q3': "Finally, write a concise description of the image and the anomalies you observed."
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
    
    def inference_multi_turn(self, image: Image.Image, prompts: Dict[str, str]) -> List[str]:
        """Perform multi-turn inference following the conversation flow."""
        responses = []
        
        # Q1: Get bounding boxes
        q1_response = self.inference_single_turn(image, prompts['Q1'])
        responses.append(q1_response)
        
        # Parse bboxes for Q2 questions
        try:
            # Try to parse bboxes from Q1 response
            json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
            match = re.search(json_pattern, q1_response, re.DOTALL)
            
            if match:
                bboxes_json = match.group(1)
                bboxes = json.loads(bboxes_json)
            else:
                bboxes = json.loads(q1_response)
            
            if isinstance(bboxes, list) and bboxes:
                # Q2: Ask about each bbox
                for bbox in bboxes:
                    bbox_str = json.dumps(bbox)
                    q2_prompt = prompts['Q2_template'].format(bbox_str=bbox_str)
                    q2_response = self.inference_single_turn(image, q2_prompt)
                    responses.append(q2_response)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse Q1 bbox response for Q2 questions")
        
        # Q3: Get caption
        q3_response = self.inference_single_turn(image, prompts['Q3'])
        responses.append(q3_response)
        
        return responses


def run_multi_turn_loc_exp_evaluation(args):
    """Run evaluation for multi-turn localization + explanation format."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"multi_turn_loc_exp_{args.dataset}_{exp_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Multi-Turn Localization + Explanation Evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    
    # Initialize model
    evaluator_model = MultiTurnLocExpEvaluator(args.exp_dir, args.device)
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
    
    # Get prompts
    prompts = evaluator_model.get_prompts()
    
    # Initialize metrics
    all_results = []
    all_iou_scores = []
    all_pixel_f1_scores = []
    all_pixel_precision_scores = []
    all_pixel_recall_scores = []
    all_rouge_l_scores = []
    all_css_scores = []
    
    total_processed = 0
    total_samples = args.max_samples if args.max_samples else len(data_iterator)
    pbar = tqdm(total=total_samples, desc="Evaluating samples", unit="sample")
    
    for gt, image_path in data_iterator:
        if args.max_samples and total_processed >= args.max_samples:
            break
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Perform multi-turn inference
            raw_outputs = evaluator_model.inference_multi_turn(image, prompts)
            
            # Process output
            predictions = process_multi_turn_loc_exp_output(
                raw_outputs, image.size[0], image.size[1]
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
                'image_path': str(image_path),
                'ground_truth': gt,
                'predictions': predictions,
                'raw_outputs': raw_outputs,
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
            logger.error(f"Error processing sample {image_path}: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(1)
            total_processed += 1
    
    pbar.close()
    
    # Calculate final metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("MULTI-TURN LOCALIZATION + EXPLANATION EVALUATION RESULTS")
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
    results_file = results_dir / f"multi_turn_loc_exp_{args.dataset}_{exp_name}_{timestamp}.json"
    
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
            'max_samples': args.max_samples,
            'format': 'multi_turn_loc_exp',
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
    parser = argparse.ArgumentParser(description="Multi-Turn Localization + Explanation Evaluation")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=["ours", "synthscars", "synartifact", "loki", "richhf", "val"], help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    run_multi_turn_loc_exp_evaluation(args)


if __name__ == "__main__":
    main()
