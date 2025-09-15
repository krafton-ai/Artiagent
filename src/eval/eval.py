"""
Main evaluation script for artifact detection models.

This script evaluates VLM/MLLM models on their ability to detect
and describe visual artifacts in images across different datasets.
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image  # type: ignore
from pathlib import Path

from models import QwenEval, InternEval, GPTEval, GeminiEval, PalEval, DiffEval, LegionEval
from eval_utils import Evaluation, parse_json, create_prompt
import legion_eval_utils
import wsol_eval_utils

def extract_bboxes(text: str) -> List[List[int]]:
    """
    Extracts all 4-number bounding boxes that appear in the form:
        [x1, y1, x2, y2]: <optional description>
    Returns a list of [x1, y1, x2, y2]. If none are found, returns [].
    """
    # Matches a bracketed list of four integers (allowing spaces) immediately followed by a colon.
    # Supports negative numbers just in case: -?\d+
    pattern = re.compile(
        r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*:',
        flags=re.UNICODE
    )

    bboxes = []
    for x1, y1, x2, y2 in pattern.findall(text):
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    return bboxes


def process_finetuned_output(raw_output: str, eval_type: str) -> Dict[str, Any]:
    """
    Process finetuned model output and format it according to the evaluation type.
    
    Args:
        raw_output: Raw text output from the finetuned model
        eval_type: Type of evaluation ('binary', 'localization', 'explanation')
        
    Returns:
        Dictionary formatted for the specific evaluation type
    """
    # Check if the model says there are no artifacts
    if "there are no artifacts in the image" in raw_output.lower():
        if eval_type == 'binary':
            return {'prediction': False}
        elif eval_type == 'localization':
            return []
        elif eval_type == 'explanation':
            return {"explanation": raw_output}
    
    if "true" in raw_output.lower():
        if eval_type == 'binary':
            return{'prediction': True}
            
    # Extract bounding boxes from the output
    bboxes = extract_bboxes(raw_output)
    
    # Process based on evaluation type
    if eval_type == 'binary':
        # If there are bboxes, there are artifacts
        return {'prediction': len(bboxes) > 0}
    
    elif eval_type == 'localization':
        # Format bboxes for localization evaluation
        bbox_list = []
        for bbox in bboxes:
            bbox_list.append({"bbox_2d": bbox})
        return bbox_list
    
    elif eval_type == 'explanation':
        # Extract explanation text (everything before the bbox list starts)
        # Find the first occurrence of a bbox pattern to split the text
        bbox_pattern = re.compile(r'\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*:')
        match = bbox_pattern.search(raw_output)
        
        if match:
            explanation_text = raw_output[:match.start()].strip()
        else:
            explanation_text = raw_output.strip()
            
        return {"explanation": explanation_text}
    
    # Fallback to raw output
    return {"raw_response": raw_output}


def setup_logging(output_dir: str, dataset_type: str, model: str, use_finetuned: bool, eval_type: str, finetune_mode: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        dataset_type: Dataset name for log file naming
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_finetuned:
        if eval_type == 'localization':
            log_file = log_dir / f'artifact_eval_{dataset_type}_{model}_finetuned_bbox_{finetune_mode}.log'
        else:
            log_file = log_dir / f'artifact_eval_{dataset_type}_{model}_finetuned_{eval_type}_{finetune_mode}.log'
    else:
        if eval_type == 'localization':
            log_file = log_dir / f'artifact_eval_{dataset_type}_{model}_bbox_{timestamp}.log'
        else:
            log_file = log_dir / f'artifact_eval_{dataset_type}_{model}_{eval_type}_{timestamp}.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class DatasetIterator:
    """
    Iterator for processing different artifact detection datasets.
    
    Supports SynthScars, SynArtifact, and LOKI datasets with
    unified interface for batch processing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset iterator.
        
        Args:
            config: Configuration dictionary with dataset settings
        """
        self.base_dir = Path(config['base_dir'])
        self.dataset_type = config['dataset_type']
        self.logger = logging.getLogger(__name__)
        
        # Initialize dataset-specific iterator
        if self.dataset_type == "synthscars":
            self._load_synthscars()
        elif self.dataset_type == "synartifact":
            self._load_synartifact()
        elif self.dataset_type == "loki":
            self._load_loki()
        elif self.dataset_type == "richhf":
            self._load_richhf()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        self.current_idx = 0
        self.total_samples = len(self.data)
        self.logger.info(f"Loaded {self.total_samples} samples from {self.dataset_type}")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples
    
    def __iter__(self):
        """Make iterator iterable."""
        self.current_idx = 0
        return self
    
    def __next__(self) -> Tuple[Dict, Path]:
        """
        Get next sample from dataset.
        
        Returns:
            Tuple of (annotation_data, image_path)
            
        Raises:
            StopIteration: When all samples have been processed
        """
        if self.current_idx >= self.total_samples:
            raise StopIteration
        
        sample = self.data[self.current_idx]
        self.current_idx += 1
        
        return self._process_sample(sample)
    
    def get_sample(self, idx: int) -> Tuple[Dict, Path]:
        """
        Get specific sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (annotation_data, image_path)
        """
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for {self.total_samples} samples")
        
        sample = self.data[idx]
        return self._process_sample(sample)
    
    def _process_sample(self, sample) -> Tuple[Dict, Path]:
        """Process a single sample based on dataset type."""
        if self.dataset_type == "synthscars":
            image_id, json_data = next(iter(sample.items()))
            image_dir = self.base_dir / "images"
            image_path = image_dir / json_data["img_file_name"]
            return json_data, image_path
            
        elif self.dataset_type == "synartifact":
            root_folder = sample.split('/')[0]
            image_id = Path(sample).stem
            
            image_path = self.base_dir / sample
            json_file = f"{root_folder}/annotation_json_artifacts_class/{image_id}.json"
            json_path = self.base_dir / json_file
            
            with open(json_path, "r") as f:
                json_data = json.load(f)
            
            return json_data, image_path
            
        elif self.dataset_type == "loki":
            json_data = sample
            image_path = self.base_dir / json_data["image_path"]
            return json_data, image_path

        elif self.dataset_type == "richhf":
            json_data = sample
            image_path = self.base_dir / json_data["filename"]
            return json_data, image_path
        # Should not reach here
        raise RuntimeError("Unsupported dataset type in _process_sample")
    
    def _load_synthscars(self):
        """Load SynthScars dataset."""
        json_path = self.base_dir / "annotations" / "test.json"
        with open(json_path, "rb") as f:
            self.data = json.load(f)
    
    def _load_synartifact(self):
        """Load SynArtifact dataset."""
        eval_set = self.base_dir / "eval.txt"
        self.data = []
        with open(eval_set, "r") as f:
            for line in f:
                self.data.append(line.strip())
    
    def _load_loki(self):
        """Load LOKI dataset."""
        json_path = self.base_dir / "open_ended_vqa.json"
        with open(json_path, "rb") as f:
            self.data = json.load(f)

    def _load_richhf(self):
        """Load RichHF-18K dataset from TFRecord file."""
        json_path = os.path.join(self.base_dir, "test.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)

def create_model(config: Dict):
    """Create model instance based on configuration."""
    model_type = config.get('model_type', 'qwen')
    
    if model_type == 'qwen':
        return QwenEval(config)
    elif model_type == 'intern':
        return InternEval(config)
    elif model_type == 'gpt':
        return GPTEval(config)
    elif model_type == 'gemini':
        return GeminiEval(config)
    elif model_type == 'pal':
        return PalEval(config)
    elif model_type == 'diff':
        return DiffEval(config)
    elif model_type == 'legion':
        return LegionEval(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def unified_inference(model, image: Image.Image, prompt: str) -> Dict[str, Any]:
    """
    Unified inference wrapper that handles different model signatures and return types.
    
    Args:
        model: Model instance 
        image: PIL Image to analyze
        prompt: Text prompt for inference
        
    Returns:
        Dictionary containing standardized inference results
    """
    try:
        # Handle models that only take image (no prompt)
        if isinstance(model, (PalEval, DiffEval, LegionEval)):
            result = model.inference(image)
        else:
            # Models that take both image and prompt
            result = model.inference(image, prompt)
        
        # Handle None returns (GPTEval can return None on error)
        if result is None:
            return {"error": "model_returned_none", "raw_response": ""}
        
        # Handle string returns (QwenEval returns string)
        if isinstance(result, str):
            try:
                # Try to parse the string as JSON
                parsed_result = parse_json(result)
                return {"parsed_output": parsed_result, "raw_response": result}
            except Exception as e:
                # If parsing fails, return as raw response
                return {"error": "json_parse_failed", "raw_response": result, "parse_error": str(e)}
        
        # Handle dictionary returns (most models)
        if isinstance(result, dict):
            # Check if it already has the expected structure
            if "raw_response" in result or "error" in result or "heatmap" in result:
                return result
            else:
                # Wrap in standard format
                return {"parsed_output": result, "raw_response": str(result)}
        
        # Handle other types
        return {"error": "unexpected_return_type", "raw_response": str(result), "type": type(result).__name__}
        
    except Exception as e:
        return {"error": "inference_exception", "raw_response": "", "exception": str(e)}


def unified_batch_inference(model, images: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    """
    Unified batch inference wrapper that handles different model signatures.
    
    Args:
        model: Model instance
        images: List of PIL Images to analyze  
        prompt: Text prompt for inference
        
    Returns:
        List of dictionaries containing standardized inference results
    """
    # try:
    # Handle models that only take images (no prompt)
    if isinstance(model, (PalEval, DiffEval, LegionEval)):
        if hasattr(model, 'inference_batch'):
            results = model.inference_batch(images)
        else:
            results = [model.inference(img) for img in images]
    else:
        # Models that take both images and prompt
        if hasattr(model, 'inference_batch'):
            results = model.inference_batch(images, prompt)
            print(results)
        else:
            results = [model.inference(img, prompt) for img in images]
    
    # Standardize each result
    standardized_results = []
    for result in results:
        if result is None:
            standardized_results.append({"error": "model_returned_none", "raw_response": ""})
        elif isinstance(result, str):
            try:
                parsed_result = parse_json(result)
                standardized_results.append({"parsed_output": parsed_result, "raw_response": result})
            except Exception as e:
                standardized_results.append({"error": "json_parse_failed", "raw_response": result, "parse_error": str(e)})
        elif isinstance(result, dict):
            if "raw_response" in result or "error" in result or "heatmap" in result:
                standardized_results.append(result)
            else:
                standardized_results.append({"parsed_output": result, "raw_response": str(result)})
        else:
            standardized_results.append({"error": "unexpected_return_type", "raw_response": str(result), "type": type(result).__name__})
    
    return standardized_results
        
    # except Exception as e:
    #     # Return error for all images
    #     return [{"error": "batch_inference_exception", "raw_response": "", "exception": str(e)} for _ in images]


def extract_prediction_result(unified_result: Dict[str, Any], use_finetuned: bool = False, eval_type: str = 'explanation') -> Dict[str, Any]:
    """
    Extract the final prediction from unified inference result for evaluation.
    
    Args:
        unified_result: Result from unified_inference or unified_batch_inference
        use_finetuned: Whether to use finetuned model output processing
        eval_type: Type of evaluation ('binary', 'localization', 'explanation')
        
    Returns:
        Dictionary suitable for evaluation
    """
    # If there's an error, return empty result with error info
    if "error" in unified_result:
        # Handle finetuned model outputs with special processing
        if use_finetuned:
            raw_response = unified_result.get("raw_response", "")
            if raw_response:
                return process_finetuned_output(raw_response, eval_type)
        else:
            return {
                "error": unified_result["error"],
                "raw_response": unified_result.get("raw_response", "")
            }
    
    # If there's a heatmap (PAL/DiffDoctor models), return it
    if "heatmap" in unified_result:
        return unified_result
    
    
    
    # If there's parsed output, use it
    if "parsed_output" in unified_result:
        return unified_result["parsed_output"]
    
    # Fallback to the whole result
    return unified_result


def run_evaluation(config: Dict, max_samples: Optional[int] = None):
    """
    Run evaluation on dataset.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    eval_type = config['eval_type']
    
    logger.info(f"Starting evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = create_model(config)
    data_iterator = DatasetIterator(config)
    evaluator = Evaluation()
    
    # Create additional evaluators for comprehensive localization evaluation
    legion_evaluator = legion_eval_utils.Evaluation()
    wsol_evaluator = wsol_eval_utils.Evaluation()

    if config.get('model_type') == 'pal' and isinstance(model, PalEval):
        memory_info = model.get_gpu_memory_info()
        for device, info in memory_info.items():
            logger.info(f"📊 GPU {device} - Allocated: {info['allocated_gb']:.2f}GB, Reserved: {info['cached_gb']:.2f}GB")
    
    # Determine number of samples to process
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    logger.info(f"Processing {total_samples} samples")

    results = {}

    prompt = create_prompt(eval_type)
    logger.info(f"Input query: {prompt}")
    
    # Process samples
    for i, (json_data, image_path) in enumerate(data_iterator):
        if max_samples is not None and i >= max_samples:
            break
            
        try:
            logger.info(f"Processing sample {i+1}/{total_samples}: {image_path}")

            # Load and process image
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            image = Image.open(str(image_path)).convert("RGB")
            if dataset_type == 'richhf':
                image = image.resize((512, 512), Image.LANCZOS)

            # Run model inference with unified interface
            unified_output = unified_inference(model, image, prompt)
            print(f"Unified output: {unified_output}")
            
            prediction = extract_prediction_result(unified_output, config['use_finetuned'], eval_type)
            print(f"Extracted prediction: {prediction}")
            # Evaluate results
            stats = evaluator.generate_statistics(
                dataset_type, eval_type, json_data, prediction, image_size=image.size
            )
            
            # For localization evaluation, also run LEGION and WSOL methods
            if eval_type == 'localization':
                legion_stats = legion_evaluator.generate_statistics(
                    dataset_type, eval_type, json_data, prediction, image_size=image.size
                )
                wsol_stats = wsol_evaluator.generate_statistics(
                    dataset_type, eval_type, json_data, prediction, image_size=image.size
                )
                
                # Merge stats with prefixes to distinguish evaluation methods
                stats.update({f'legion_{k}': v for k, v in legion_stats.items() if k not in ['binary_success', 'rouge_l', 'css', 'classification', 'has_gt_artifacts', 'has_pred_artifacts']})
                stats.update({f'wsol_{k}': v for k, v in wsol_stats.items() if k not in ['binary_success', 'rouge_l', 'css', 'classification', 'has_gt_artifacts', 'has_pred_artifacts']})
            if eval_type == 'binary':
                sample_result = {
                    'image_path': str(image_path),
                    'binary_success': stats['binary_success'],
                    'classification': stats['classification'],
                    'has_gt_artifacts': stats['has_gt_artifacts'],
                    'has_pred_artifacts': stats['has_pred_artifacts'],
                    'prediction': prediction
                }
                logger.info(
                    f"Sample {i + 1} - Binary: {sample_result['binary_success']}, "
                    f"Prediction: {prediction}"
                )
            elif eval_type == 'localization':
                sample_result = {
                    'image_path': str(image_path),
                    # Standard evaluation metrics
                    'iou': stats['iou'],
                    'loc_tp': stats['loc_tp'],
                    'loc_fp': stats['loc_fp'],
                    'loc_fn': stats['loc_fn'],
                    'loc_precision': stats['loc_precision'],
                    'loc_recall': stats['loc_recall'],
                    'loc_f1': stats['loc_f1'],
                    # LEGION evaluation metrics
                    'legion_iou': stats.get('legion_iou'),
                    'legion_miou': stats.get('legion_miou'),
                    'legion_iou_foreground': stats.get('legion_iou_foreground'),
                    'legion_iou_background': stats.get('legion_iou_background'),
                    'legion_pixel_f1': stats.get('legion_pixel_f1'),
                    'legion_pixel_precision': stats.get('legion_pixel_precision'),
                    'legion_pixel_recall': stats.get('legion_pixel_recall'),
                    # WSOL evaluation metrics
                    'wsol_iou': stats.get('wsol_iou'),
                    'prediction': prediction
                }
                if sample_result.get('iou', None) is None:
                    logger.info(f"Sample {i + 1} - Skipped (negative sample)")
                else:
                    logger.info(
                        f"Sample {i + 1} - IoU: {sample_result['iou']:.3f}, "
                        f"F1: {sample_result['loc_f1']:.3f} (P: {sample_result['loc_precision']:.3f}, "
                        f"R: {sample_result['loc_recall']:.3f}, TP/FP/FN: {sample_result['loc_tp']}/{sample_result['loc_fp']}/{sample_result['loc_fn']})"
                    )
            elif eval_type == 'explanation':
                sample_result = {
                    'image_path': str(image_path),
                    'rouge_l': stats['rouge_l'],
                    'css': stats['css'],
                    'prediction': prediction
                }
                logger.info(
                    f"Sample {i + 1} - ROUGE-L: {sample_result['rouge_l']:.3f}, "
                    f"CSS: {sample_result['css']:.3f}"
                )
            else:
                raise ValueError(f"Unsupported evaluation type: {eval_type}")

            # Store results
            results[i] = sample_result
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            continue

    logger.info("Evaluation completed!")
    if results:
        # Initialize all variables
        binary_accuracy = 0.0
        f1_metrics = {}
        mean_iou = 0.0
        mean_rouge_l = 0.0
        mean_css = 0.0
        valid_loc_results = []
        mean_loc_f1 = 0.0
        mean_loc_precision = 0.0
        mean_loc_recall = 0.0
        total_loc_tp = 0
        total_loc_fp = 0
        total_loc_fn = 0
        
        if eval_type == 'binary':
            binary_accuracy = sum(r.get('binary_success', False) for _, r in results.items()) / total_samples
            # Compute F1 metrics
            f1_metrics = evaluator.compute_f1_metrics(results)
        elif eval_type == 'localization':
            # Filter out None values for SynArtifact negative samples
            valid_loc_results = [r for _, r in results.items() if r.get('iou') is not None]
            mean_iou = sum(r.get('iou', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_f1 = sum(r.get('loc_f1', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_precision = sum(r.get('loc_precision', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_recall = sum(r.get('loc_recall', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            total_loc_tp = sum(r.get('loc_tp', 0) for r in valid_loc_results if r.get('loc_tp') is not None)
            total_loc_fp = sum(r.get('loc_fp', 0) for r in valid_loc_results if r.get('loc_fp') is not None)
            total_loc_fn = sum(r.get('loc_fn', 0) for r in valid_loc_results if r.get('loc_fn') is not None)
            
            # LEGION evaluation metrics
            legion_valid_results = [r for r in valid_loc_results if r.get('legion_iou') is not None]
            legion_mean_iou = sum(r.get('legion_iou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_miou = sum(r.get('legion_miou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_fg = sum(r.get('legion_iou_foreground', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_bg = sum(r.get('legion_iou_background', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_f1 = sum(r.get('legion_pixel_f1', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_precision = sum(r.get('legion_pixel_precision', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_recall = sum(r.get('legion_pixel_recall', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            
            # WSOL evaluation metrics
            wsol_valid_results = [r for r in valid_loc_results if r.get('wsol_iou') is not None]
            wsol_mean_iou = sum(r.get('wsol_iou', 0.0) for r in wsol_valid_results) / len(wsol_valid_results) if wsol_valid_results else 0.0
        elif eval_type == 'explanation':
            mean_rouge_l = sum(r.get('rouge_l', 0.0) for _, r in results.items()) / total_samples
            mean_css = sum(r.get('css', 0.0) for _, r in results.items()) / total_samples
        
        logger.info("=" * 60)
        logger.info("BATCH EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_samples}")
        if eval_type == 'binary':
            logger.info(f"Binary classification accuracy: {binary_accuracy:.3f}")
            logger.info("")
            logger.info("F1 METRICS:")
            logger.info(f"  TP: {f1_metrics.get('tp', 0)}, FP: {f1_metrics.get('fp', 0)}, FN: {f1_metrics.get('fn', 0)}, TN: {f1_metrics.get('tn', 0)}")
            logger.info(f"  Precision: {f1_metrics.get('precision', 0.0):.3f}")
            logger.info(f"  Recall: {f1_metrics.get('recall', 0.0):.3f}")
            logger.info(f"  F1-Score: {f1_metrics.get('f1_positive', 0.0):.3f}")
            logger.info(f"  Negative Precision: {f1_metrics['precision_negative']:.3f}")
            logger.info(f"  Negative Recall: {f1_metrics['recall_negative']:.3f}")
            logger.info(f"  Negative F1-Score: {f1_metrics['f1_negative']:.3f}")
            logger.info(f"  Macro F1: {f1_metrics['macro_f1']:.3f}")
            logger.info(f"  Accuracy: {f1_metrics.get('accuracy', 0.0):.3f}")
            logger.info("")
        elif eval_type == 'localization':
            valid_samples = len(valid_loc_results)
            logger.info("=" * 80)
            logger.info("COMPREHENSIVE LOCALIZATION EVALUATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Valid samples (positive samples): {valid_samples}")
            logger.info("")
            
            # Standard evaluation results
            logger.info("📊 STANDARD EVALUATION (Threshold-Independent Bbox Metrics):")
            logger.info(f"  Mean IoU: {mean_iou:.3f}")
            logger.info(f"  Mean F1: {mean_loc_f1:.3f}")
            logger.info(f"  Mean Precision: {mean_loc_precision:.3f}")
            logger.info(f"  Mean Recall: {mean_loc_recall:.3f}")
            logger.info(f"  Total TP/FP/FN: {total_loc_tp}/{total_loc_fp}/{total_loc_fn}")
            
            # Compute global F1 (across all samples)
            global_precision = total_loc_tp / (total_loc_tp + total_loc_fp) if (total_loc_tp + total_loc_fp) > 0 else 0.0
            global_recall = total_loc_tp / (total_loc_tp + total_loc_fn) if (total_loc_tp + total_loc_fn) > 0 else 0.0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
            logger.info(f"  Global Precision: {global_precision:.3f}")
            logger.info(f"  Global Recall: {global_recall:.3f}")
            logger.info(f"  Global F1: {global_f1:.3f}")
            logger.info("")
            
            # LEGION evaluation results
            legion_samples = len(legion_valid_results)
            logger.info("🎯 LEGION EVALUATION (Pixel-Level Segmentation Metrics):")
            logger.info(f"  Valid samples: {legion_samples}")
            logger.info(f"  Mean IoU (Legacy): {legion_mean_iou:.3f}")
            logger.info(f"  Mean IoU (mIoU): {legion_mean_miou:.3f}")
            logger.info(f"    - Foreground IoU: {legion_mean_iou_fg:.3f}")
            logger.info(f"    - Background IoU: {legion_mean_iou_bg:.3f}")
            logger.info(f"  Pixel F1 Score: {legion_mean_pixel_f1:.3f}")
            logger.info(f"  Pixel Precision: {legion_mean_pixel_precision:.3f}")
            logger.info(f"  Pixel Recall: {legion_mean_pixel_recall:.3f}")
            logger.info("")
            
            # WSOL evaluation results
            wsol_samples = len(wsol_valid_results)
            logger.info("🔄 WSOL EVALUATION (Threshold-Independent IoU):")
            logger.info(f"  Valid samples: {wsol_samples}")
            logger.info(f"  Mean IoU: {wsol_mean_iou:.3f}")
            logger.info("")
        elif eval_type == 'explanation':
            logger.info(f"Mean ROUGE-L (all samples): {mean_rouge_l:.3f}")
            logger.info(f"Mean CSS (all samples): {mean_css:.3f}")

    # Final GPU memory reporting for PAL model
    if config.get('model_type') == 'pal' and isinstance(model, PalEval):
        logger.info("📊 Final GPU Memory Usage:")
        memory_info = model.get_gpu_memory_info()
        for device, info in memory_info.items():
            logger.info(f"    {device} - Max Allocated: {info['max_allocated_gb']:.2f}GB, Current: {info['allocated_gb']:.2f}GB")
        
        # Final cache clearing
        model.clear_gpu_cache()

    if isinstance(model, GPTEval):
        try:
            logger.info(f"Total cost: {model.money_manager.total_cost}")
        except Exception:
            pass
    
    return results

def run_batch_evaluation(config: Dict, max_samples: Optional[int] = None):
    """
    Run evaluation on multiple images with optional visualization.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to process
        enable_visualization: Whether to save visualization results
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    eval_type = config['eval_type']

    # Setup logging
    logger.info(f"Starting evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    logger.info(f"Starting batch evaluation for {dataset_type} dataset")
    if max_samples:
        logger.info(f"Processing {max_samples} samples")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = create_model(config)
    data_iterator = DatasetIterator(config)
    evaluator = Evaluation()
    
    # Create additional evaluators for comprehensive localization evaluation
    legion_evaluator = legion_eval_utils.Evaluation()
    wsol_evaluator = wsol_eval_utils.Evaluation()

    if config.get('model_type') == 'pal' and isinstance(model, PalEval):
        memory_info = model.get_gpu_memory_info()
        for device, info in memory_info.items():
            logger.info(f"📊 GPU {device} - Allocated: {info['allocated_gb']:.2f}GB, Reserved: {info['cached_gb']:.2f}GB")
    
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    results = {}
    processed = 0

    prompt = create_prompt(eval_type)
    logger.info(f"Input query: {prompt}")

    # Determine batch size: default to 2 if not provided in config
    target_batch_size: int = int(config.get('batch_size', 2) or 2)
    current_batch_size: int = target_batch_size

    try:
        while True:
            if max_samples and processed >= max_samples:
                break
            # Collect a batch of samples
            batch_json_data: List[Dict[str, Any]] = []
            batch_image_paths: List[Path] = []
            batch_images: List[Image.Image] = []
            while len(batch_images) < current_batch_size:
                try:
                    json_data, image_path = next(data_iterator)
                except StopIteration:
                    break
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    continue
                batch_json_data.append(json_data)
                batch_image_paths.append(image_path)
                batch_images.append(image)
                if max_samples and (processed + len(batch_images)) >= max_samples:
                    break
            if not batch_images:
                # No more data
                break
            logger.info(
                f"Processing batch starting at index {processed + 1} with size {len(batch_images)}"
            )
            # Run batched inference with OOM fallback
            try:
                # Use unified batch inference
                batch_unified_results = unified_batch_inference(model, batch_images, prompt)
                batch_results = [extract_prediction_result(result, config['use_finetuned'], eval_type) for result in batch_unified_results]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and len(batch_images) > 1:
                    logger.warning("OOM during batched inference. Falling back to per-sample inference for this batch.")
                    # Reduce future batch size to be more conservative
                    current_batch_size = max(1, current_batch_size // 2)
                    batch_results = []
                    for img in batch_images:
                        try:
                            unified_result = unified_inference(model, img, prompt)
                            batch_results.append(extract_prediction_result(unified_result, config['use_finetuned'], eval_type))
                        except Exception as inner_e:
                            logger.error(f"Per-sample inference failed: {inner_e}")
                            batch_results.append({
                                'error': str(inner_e)
                            })
                else:
                    raise
            # Evaluate each item in the batch
            for idx, (json_data, image_path, image, result) in enumerate(
                zip(batch_json_data, batch_image_paths, batch_images, batch_results)
            ):
                stats = evaluator.generate_statistics(
                    dataset_type, eval_type, json_data, result, image_size=image.size
                )
                
                # For localization evaluation, also run LEGION and WSOL methods
                if eval_type == 'localization':
                    legion_stats = legion_evaluator.generate_statistics(
                        dataset_type, eval_type, json_data, result, image_size=image.size
                    )
                    wsol_stats = wsol_evaluator.generate_statistics(
                        dataset_type, eval_type, json_data, result, image_size=image.size
                    )
                    
                    # Merge stats with prefixes to distinguish evaluation methods
                    stats.update({f'legion_{k}': v for k, v in legion_stats.items() if k not in ['binary_success', 'rouge_l', 'css', 'classification', 'has_gt_artifacts', 'has_pred_artifacts']})
                    stats.update({f'wsol_{k}': v for k, v in wsol_stats.items() if k not in ['binary_success', 'rouge_l', 'css', 'classification', 'has_gt_artifacts', 'has_pred_artifacts']})
                if eval_type == 'binary':
                    sample_result = {
                        'image_path': str(image_path),
                        'binary_success': stats['binary_success'],
                        'classification': stats['classification'],
                        'has_gt_artifacts': stats['has_gt_artifacts'],
                        'has_pred_artifacts': stats['has_pred_artifacts'],
                        'prediction': result
                    }
                    logger.info(
                        f"Sample {processed + idx + 1} - Binary: {sample_result['binary_success']}, "
                        f"Prediction: {result}"
                    )
                elif eval_type == 'localization':
                    sample_result = {
                        'image_path': str(image_path),
                        # Standard evaluation metrics
                        'iou': stats['iou'],
                        'loc_tp': stats['loc_tp'],
                        'loc_fp': stats['loc_fp'],
                        'loc_fn': stats['loc_fn'],
                        'loc_precision': stats['loc_precision'],
                        'loc_recall': stats['loc_recall'],
                        'loc_f1': stats['loc_f1'],
                        # LEGION evaluation metrics
                        'legion_iou': stats.get('legion_iou'),
                        'legion_miou': stats.get('legion_miou'),
                        'legion_iou_foreground': stats.get('legion_iou_foreground'),
                        'legion_iou_background': stats.get('legion_iou_background'),
                        'legion_pixel_f1': stats.get('legion_pixel_f1'),
                        'legion_pixel_precision': stats.get('legion_pixel_precision'),
                        'legion_pixel_recall': stats.get('legion_pixel_recall'),
                        # WSOL evaluation metrics
                        'wsol_iou': stats.get('wsol_iou'),
                        'prediction': result
                    }
                    if sample_result.get('iou', None) is None:
                        logger.info(f"Sample {processed + idx + 1} - Skipped (negative sample)")
                    else:
                        logger.info(
                            f"Sample {processed + idx + 1} - IoU: {sample_result['iou']:.3f}, "
                            f"F1: {sample_result['loc_f1']:.3f} (P: {sample_result['loc_precision']:.3f}, "
                            f"R: {sample_result['loc_recall']:.3f}, TP/FP/FN: {sample_result['loc_tp']}/{sample_result['loc_fp']}/{sample_result['loc_fn']})"
                        )
                elif eval_type == 'explanation':
                    sample_result = {
                        'image_path': str(image_path),
                        'rouge_l': stats['rouge_l'],
                        'css': stats['css'],
                        'prediction': result
                    }
                    logger.info(
                        f"Sample {processed + idx + 1} - ROUGE-L: {sample_result['rouge_l']:.3f}, "
                        f"CSS: {sample_result['css']:.3f}"
                    )
                else:
                    raise ValueError(f"Unsupported evaluation type: {eval_type}")

                results[processed + idx] = sample_result

            processed += len(batch_images)

            if config.get('model_type') == 'pal' and isinstance(model, PalEval) and processed % (10 * current_batch_size) == 0:
                model.clear_gpu_cache()
                logger.info(f"🧹 Cleared GPU cache after processing {processed} samples")
            
    except StopIteration:
        logger.info("Reached end of dataset")
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
    
    # Compute summary statistics
    if results:
        total_samples = len(results)
        binary_accuracy = 0.0
        f1_metrics = {}
        mean_iou = 0.0
        mean_loc_f1 = 0.0
        mean_loc_precision = 0.0
        mean_loc_recall = 0.0
        total_loc_tp = 0
        total_loc_fp = 0
        total_loc_fn = 0
        mean_rouge_l = 0.0
        mean_css = 0.0
        valid_loc_results = []
        
        if eval_type == 'binary':
            binary_accuracy = sum(r.get('binary_success', False) for _, r in results.items()) / total_samples
            # Compute F1 metrics
            f1_metrics = evaluator.compute_f1_metrics(results)
        elif eval_type == 'localization':
            # Filter out None values for SynArtifact negative samples
            valid_loc_results = [r for _, r in results.items() if r.get('iou') is not None]
            mean_iou = sum(r.get('iou', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_f1 = sum(r.get('loc_f1', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_precision = sum(r.get('loc_precision', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_recall = sum(r.get('loc_recall', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            total_loc_tp = sum(r.get('loc_tp', 0) for r in valid_loc_results if r.get('loc_tp') is not None)
            total_loc_fp = sum(r.get('loc_fp', 0) for r in valid_loc_results if r.get('loc_fp') is not None)
            total_loc_fn = sum(r.get('loc_fn', 0) for r in valid_loc_results if r.get('loc_fn') is not None)
            
            # LEGION evaluation metrics
            legion_valid_results = [r for r in valid_loc_results if r.get('legion_iou') is not None]
            legion_mean_iou = sum(r.get('legion_iou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_miou = sum(r.get('legion_miou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_fg = sum(r.get('legion_iou_foreground', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_bg = sum(r.get('legion_iou_background', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_f1 = sum(r.get('legion_pixel_f1', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_precision = sum(r.get('legion_pixel_precision', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_recall = sum(r.get('legion_pixel_recall', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            
            # WSOL evaluation metrics
            wsol_valid_results = [r for r in valid_loc_results if r.get('wsol_iou') is not None]
            wsol_mean_iou = sum(r.get('wsol_iou', 0.0) for r in wsol_valid_results) / len(wsol_valid_results) if wsol_valid_results else 0.0
        elif eval_type == 'explanation':
            mean_rouge_l = sum(r.get('rouge_l', 0.0) for _, r in results.items()) / total_samples
            mean_css = sum(r.get('css', 0.0) for _, r in results.items()) / total_samples
        
        logger.info("=" * 60)
        logger.info("BATCH EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_samples}")
        if eval_type == 'binary':
            logger.info(f"Binary classification accuracy: {binary_accuracy:.3f}")
            logger.info("")
            logger.info("F1 METRICS:")
            logger.info(f"  TP: {f1_metrics['tp']}, FP: {f1_metrics['fp']}, FN: {f1_metrics['fn']}, TN: {f1_metrics['tn']}")
            logger.info(f"  Precision: {f1_metrics['precision']:.3f}")
            logger.info(f"  Recall: {f1_metrics['recall']:.3f}")
            logger.info(f"  Positive F1-Score: {f1_metrics['f1_positive']:.3f}")
            logger.info(f"  Negative Precision: {f1_metrics['precision_negative']:.3f}")
            logger.info(f"  Negative Recall: {f1_metrics['recall_negative']:.3f}")
            logger.info(f"  Negative F1-Score: {f1_metrics['f1_negative']:.3f}")
            logger.info(f"  Macro F1: {f1_metrics['macro_f1']:.3f}")
            logger.info(f"  Accuracy: {f1_metrics['accuracy']:.3f}")
            logger.info("")
        elif eval_type == 'localization':
            valid_samples = len(valid_loc_results)
            logger.info("=" * 80)
            logger.info("COMPREHENSIVE LOCALIZATION EVALUATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Valid samples (positive samples): {valid_samples}")
            logger.info("")
            
            # Standard evaluation results
            logger.info("📊 STANDARD EVALUATION (Threshold-Independent Bbox Metrics):")
            logger.info(f"  Mean IoU: {mean_iou:.3f}")
            logger.info(f"  Mean F1: {mean_loc_f1:.3f}")
            logger.info(f"  Mean Precision: {mean_loc_precision:.3f}")
            logger.info(f"  Mean Recall: {mean_loc_recall:.3f}")
            logger.info(f"  Total TP/FP/FN: {total_loc_tp}/{total_loc_fp}/{total_loc_fn}")
            
            # Compute global F1 (across all samples)
            global_precision = total_loc_tp / (total_loc_tp + total_loc_fp) if (total_loc_tp + total_loc_fp) > 0 else 0.0
            global_recall = total_loc_tp / (total_loc_tp + total_loc_fn) if (total_loc_tp + total_loc_fn) > 0 else 0.0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
            logger.info(f"  Global Precision: {global_precision:.3f}")
            logger.info(f"  Global Recall: {global_recall:.3f}")
            logger.info(f"  Global F1: {global_f1:.3f}")
            logger.info("")
            
            # LEGION evaluation results
            legion_samples = len(legion_valid_results)
            logger.info("🎯 LEGION EVALUATION (Pixel-Level Segmentation Metrics):")
            logger.info(f"  Valid samples: {legion_samples}")
            logger.info(f"  Mean IoU (Legacy): {legion_mean_iou:.3f}")
            logger.info(f"  Mean IoU (mIoU): {legion_mean_miou:.3f}")
            logger.info(f"    - Foreground IoU: {legion_mean_iou_fg:.3f}")
            logger.info(f"    - Background IoU: {legion_mean_iou_bg:.3f}")
            logger.info(f"  Pixel F1 Score: {legion_mean_pixel_f1:.3f}")
            logger.info(f"  Pixel Precision: {legion_mean_pixel_precision:.3f}")
            logger.info(f"  Pixel Recall: {legion_mean_pixel_recall:.3f}")
            logger.info("")
            
            # WSOL evaluation results
            wsol_samples = len(wsol_valid_results)
            logger.info("🔄 WSOL EVALUATION (Threshold-Independent IoU):")
            logger.info(f"  Valid samples: {wsol_samples}")
            logger.info(f"  Mean IoU: {wsol_mean_iou:.3f}")
            logger.info("")
        elif eval_type == 'explanation':
            logger.info(f"Mean ROUGE-L (all samples): {mean_rouge_l:.3f}")
            logger.info(f"Mean CSS (all samples): {mean_css:.3f}")
    
    # Final GPU memory reporting for PAL model
    if config.get('model_type') == 'pal' and isinstance(model, PalEval):
        logger.info("📊 Final GPU Memory Usage:")
        memory_info = model.get_gpu_memory_info()
        for device, info in memory_info.items():
            logger.info(f"    {device} - Max Allocated: {info['max_allocated_gb']:.2f}GB, Current: {info['allocated_gb']:.2f}GB")
        
        # Final cache clearing
        model.clear_gpu_cache()

    if isinstance(model, GPTEval):
        try:
            logger.info(f"Total cost: {model.money_manager.total_cost}")
        except Exception:
            pass

    return results

def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate VLM/MLLM models on artifact detection tasks'
    )
    parser.add_argument('--model', type=str, choices=['qwen', 'intern', 'gpt', 'gemini', 'pal', 'diff', 'legion'], 
                       default='qwen', help='Model type to evaluate (default: qwen)')
    parser.add_argument('--dataset', type=str, 
                       choices=['synthscars', 'synartifact', 'loki', 'richhf'], 
                       default='loki', help='Dataset to evaluate on (default: loki)')
    parser.add_argument('--type', type=str,
                       choices=['binary', 'localization', 'explanation'],
                       default='explanation', help='Evaluation type (default: explanation)')
    parser.add_argument('--use-finetuned', action='store_true',
                       help='Use finetuned model instead of base model')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device for inference (default: cuda:0)')
    parser.add_argument('--log-dir', type=str, default='eval_logs',
                       help='Directory for logs (default: eval_logs)')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                       help='Directory for results (default: eval_results)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Custom base directory for dataset')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batched inference size (default: 1)')
    parser.add_argument('--use-multi-gpu', action='store_true',
                       help='Enable multi-GPU inference for PAL model')
    parser.add_argument('--gpu-devices', type=str, nargs='+', default=None,
                       help='Specify GPU devices to use (e.g., 0 1 or cuda:0 cuda:1)')
    parser.add_argument('--finetune-mode', type=str, 
                       choices=['1k', '3k_all', '3k_bin', '3k_loc', '3k_exp', '3k_reasoned_bin', '3k_reasoned_loc', '8k'])
                       
    args = parser.parse_args()
    
    # Set dataset paths if not provided
    if args.base_dir is None:
        dataset_paths = {
            'synthscars': "/home/jovyan/image-artifacts/data/SynthScars/test",
            'synartifact': "/home/jovyan/image-artifacts/data/SynArtifact/data",
            'loki': "/home/jovyan/image-artifacts/data/loki",
            'richhf': "/home/jovyan/image-artifacts/data/richhf-18k"
        }
        base_dir = dataset_paths.get(args.dataset)
        if base_dir is None:
            raise ValueError(f"No default path for dataset: {args.dataset}")
    else:
        base_dir = args.base_dir
    
    # Setup configuration
    config = {
        'model_type': args.model,
        'dataset_type': args.dataset,
        'eval_type': args.type,
        'base_dir': base_dir,
        'log_dir': args.log_dir,
        'use_finetuned': args.use_finetuned,
        'device': args.device,
        'batch_size': args.batch_size,
        'use_multi_gpu': args.use_multi_gpu,
        'gpu_devices': args.gpu_devices,
        'finetune_mode': args.finetune_mode
    }
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.dataset, args.model, args.use_finetuned, args.type, args.finetune_mode)
    
    logger.info(f"🚀 Starting evaluation for {args.dataset.upper()} dataset")
    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"🤖 Finetuned: {args.use_finetuned}")
    logger.info(f"🗒️ Evaluating: {args.type}")
    logger.info(f"📁 Dataset path: {base_dir}")
    logger.info(f"🔧 Device: {args.device}")

    # Multi-GPU configuration logging
    if args.use_multi_gpu:
        if args.model != 'pal':
            logger.warning("⚠️  Multi-GPU is only supported for PAL model. Ignoring --use-multi-gpu flag.")
            config['use_multi_gpu'] = False
        else:
            logger.info(f"🔀 Multi-GPU: Enabled")
            if args.gpu_devices:
                logger.info(f"🔀 GPU devices: {args.gpu_devices}")
            else:
                logger.info(f"🔀 GPU devices: Auto-detect (using first 2 GPUs)")
    
    try:
        # Run evaluation
        if args.batch_size > 1:
            results = run_batch_evaluation(config, args.max_samples)
        else:
            results = run_evaluation(config, args.max_samples)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.use_finetuned:
            if args.type == 'localization':
                results_file = output_dir / f"results_{args.dataset}_{args.model}_finetuned_bbox_{args.finetune_mode}.json"
            else:
                results_file = output_dir / f"results_{args.dataset}_{args.model}_finetuned_{args.type}_{args.finetune_mode}.json"
        else:
            if args.type == 'localization':
                results_file = output_dir / f"results_{args.dataset}_{args.model}_bbox_{timestamp}.json"
            else:
                results_file = output_dir / f"results_{args.dataset}_{args.model}_{args.type}_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Evaluation completed! Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()