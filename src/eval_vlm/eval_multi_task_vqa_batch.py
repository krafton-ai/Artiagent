"""
Batch Evaluation for Multi-Task VQA Finetuned Models

This evaluation script tests three independent single-image tasks using separate 
prompts from the training templates:
  - Task 1.1: Binary Detection (single image)
  - Task 1.2: Localization (single image)  
  - Task 1.3: Global Explanation (single image)

Note: This does NOT evaluate:
  - Task 1.4: Regional Explanation (evaluated implicitly via other metrics)
  - Tasks 4.1-4.4: Pair-image tasks (requires two images)
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
sys.path.append(str(Path(__file__).parent.parent / "train" / "LLaMA-Factory" / "data_gen"))

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import ModelArguments, DataArguments, EvaluationArguments, get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data.mm_plugin import IMAGE_PLACEHOLDER

from eval_batch_utils import (
    DatasetIterator,
    setup_logging,
)

# Import VQA prompts
from vqa_gen.vqa_prompts import VQAPrompts

# Import from parent eval directory
sys.path.append(str(Path(__file__).parent.parent / "eval"))
from eval_utils import Evaluation
import legion_eval_utils
import wsol_eval_utils

logger = logging.getLogger(__name__)


def parse_binary_detection_response(response: str) -> Dict[str, Any]:
    """Parse binary detection JSON response.
    
    Expected format: {"type":"binary_detection","artifact_present":"yes|no"}
    
    Returns:
        Dict with 'prediction' (bool)
    """
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
        
        # Extract artifact_present field
        artifact_present = parsed.get('artifact_present', 'no').lower()
        prediction = artifact_present == 'yes'
        
        return {'prediction': prediction}
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse binary detection JSON: {e}")
        # Fallback to text-based detection
        if "yes" in response.lower() and "no" not in response.lower():
            return {'prediction': True}
        return {'prediction': False}


def parse_localization_response(response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Parse localization JSON response.
    
    Expected format: {"type":"localization","coord_space":"pixel","bboxes":[{"bbox":[xmin,ymin,xmax,ymax]}, ...]}
    
    Returns:
        List of dicts with 'bbox_2d' field
    """
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
        
        # Extract bboxes
        bboxes = parsed.get('bboxes', [])
        
        bbox_list = []
        for bbox_obj in bboxes:
            if isinstance(bbox_obj, dict) and 'bbox' in bbox_obj:
                bbox = bbox_obj['bbox']
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
        
        return bbox_list
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse localization JSON: {e}")
        return []


def parse_global_explanation_response(response: str) -> Dict[str, Any]:
    """Parse global explanation JSON response.
    
    Expected format: {"type":"global_explanation","explanations":["<short description>"]}
    
    Returns:
        Dict with 'explanation' (str)
    """
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
        
        # Extract explanations array
        explanations = parsed.get('explanations', [])
        
        if explanations:
            # Join all explanations into a single string
            explanation_text = " ".join(str(e) for e in explanations)
        else:
            explanation_text = "No artifacts detected."
        
        return {"explanation": explanation_text}
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse global explanation JSON: {e}")
        # Fallback to raw response
        return {"explanation": response if response else "No artifacts detected."}


class MultiTaskVQAEvaluator:
    """Evaluator for multi-task VQA format models."""
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
    
    def get_prompts(self) -> Dict[str, str]:
        """Get three evaluation prompts using training templates.
        
        Returns prompts for single-image tasks:
          - 1.1 Binary Detection
          - 1.2 Localization  
          - 1.3 Global Explanation
        
        Note: Adds <image> token since templates don't include it.
        """
        return {
            'binary': f"<image>\n{VQAPrompts.get_binary_detection(include_format=True)}",
            'localization': f"<image>\n{VQAPrompts.get_localization(include_format=True)}",
            'explanation': f"<image>\n{VQAPrompts.get_global_explanation(include_format=True)}"
        }
    
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
        """Perform batch inference for a single task."""
        batch_size = len(images)
        
        # Process each sample
        batch_input_ids = []
        for image in images:
            # Note: prompt already includes <image> token, so don't add IMAGE_PLACEHOLDER
            messages = [{"role": "user", "content": prompt}]
            
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


def run_multi_task_vqa_evaluation(args):
    """Run evaluation for multi-task VQA format."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"multi_task_vqa_{args.dataset}_{exp_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Multi-Task VQA Evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"📊 Evaluating three independent tasks:")
    logger.info(f"   1. Binary Detection")
    logger.info(f"   2. Localization")
    logger.info(f"   3. Global Explanation")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize model
    evaluator_model = MultiTaskVQAEvaluator(args.exp_dir, args.device)
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
    
    # Get three prompts
    prompts = evaluator_model.get_prompts()
    logger.info("\nUsing training template prompts:")
    logger.info(f"Binary prompt: {prompts['binary'][:80]}...")
    logger.info(f"Localization prompt: {prompts['localization'][:80]}...")
    logger.info(f"Explanation prompt: {prompts['explanation'][:80]}...")
    
    # Initialize metrics for all three tasks
    all_results = []
    all_binary_success = []
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
                # Run three separate inference calls for each task
                binary_outputs = evaluator_model.inference_batch(batch_images, prompts['binary'])
                loc_outputs = evaluator_model.inference_batch(batch_images, prompts['localization'])
                expl_outputs = evaluator_model.inference_batch(batch_images, prompts['explanation'])
                
                for i, (img_path, gt) in enumerate(batch_metadata):
                    image = batch_images[i]
                    
                    # Parse outputs for each task
                    binary_pred = parse_binary_detection_response(binary_outputs[i])
                    loc_pred = parse_localization_response(loc_outputs[i], image.size[0], image.size[1])
                    expl_pred = parse_global_explanation_response(expl_outputs[i])
                    
                    # Calculate stats for all three tasks
                    binary_stats = evaluator.generate_statistics(
                        args.dataset, 'binary', gt, binary_pred, image_size=image.size
                    )
                    
                    loc_stats = evaluator.generate_statistics(
                        args.dataset, 'localization', gt, loc_pred, image_size=image.size
                    )
                    
                    # Additional localization metrics
                    legion_stats = legion_evaluator.generate_statistics(
                        args.dataset, 'localization', gt, loc_pred, image_size=image.size
                    )
                    wsol_stats = wsol_evaluator.generate_statistics(
                        args.dataset, 'localization', gt, loc_pred, image_size=image.size
                    )
                    
                    expl_stats = evaluator.generate_statistics(
                        args.dataset, 'explanation', gt, expl_pred, image_size=image.size
                    )
                    
                    # Collect binary metrics
                    binary_success = binary_stats.get('binary_success', False)
                    all_binary_success.append(binary_success)
                    
                    # Determine if GT has artifacts
                    if args.dataset == 'ours':
                        has_gt = gt.get('has_artifacts', True)
                    elif args.dataset == 't2i':
                        has_gt = bool(gt.get('Artifacts annotation', []))
                    elif args.dataset == 'synartifact':
                        has_gt = bool(gt.get('Artifacts annotation', []))
                    else:
                        has_gt = True
                    
                    # Collect localization metrics (only for positive samples)
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
                    iou_val = legion_stats.get('iou', 0.0) if has_gt else 0.0
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
                        'predictions': {
                            'binary': binary_pred,
                            'localization': loc_pred,
                            'explanation': expl_pred
                        },
                        'raw_outputs': {
                            'binary': binary_outputs[i],
                            'localization': loc_outputs[i],
                            'explanation': expl_outputs[i]
                        },
                        # Binary metrics
                        'binary_success': binary_success,
                        'classification': binary_stats.get('classification'),
                        # Localization metrics
                        'iou': loc_stats.get('iou') if has_gt else None,
                        'loc_f1': loc_stats.get('loc_f1') if has_gt else None,
                        'loc_precision': loc_stats.get('loc_precision') if has_gt else None,
                        'loc_recall': loc_stats.get('loc_recall') if has_gt else None,
                        'legion_iou': legion_stats.get('iou') if has_gt else None,
                        'legion_pixel_f1': legion_stats.get('pixel_f1') if has_gt else None,
                        'legion_pixel_precision': legion_stats.get('pixel_precision') if has_gt else None,
                        'legion_pixel_recall': legion_stats.get('pixel_recall') if has_gt else None,
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
    logger.info("MULTI-TASK VQA EVALUATION RESULTS")
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
    logger.info("\nGLOBAL EXPLANATION:")
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
    results_file = results_dir / f"multi_task_vqa_{args.dataset}_{exp_name}_{timestamp}.json"
    
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
            'format': 'multi_task_vqa',
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
    parser = argparse.ArgumentParser(description="Multi-Task VQA Evaluation")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=["ours", "synthscars", "synartifact", "loki", "richhf", "val"], help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    run_multi_task_vqa_evaluation(args)


if __name__ == "__main__":
    main()

