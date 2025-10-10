"""
Unified evaluation script for all artifact detection tasks.

This script evaluates VLM/MLLM models on their ability to simultaneously perform:
1. Binary classification (artifact presence)
2. Localization (bounding box detection)
3. Explanation (artifact description)

All tasks are evaluated from a single model inference.
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
from eval_utils import Evaluation, parse_json
import legion_eval_utils
import wsol_eval_utils


def extract_bboxes(text):
    """
    Extract all bounding boxes of the format [int, int, int, int] from a string.

    Args:
        text (str): The input string containing potential bounding boxes.

    Returns:
        List of tuples, each containing 4 integers representing a bounding box.
    """
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, text)
    bboxes = [list(map(int, match)) for match in matches]
    return bboxes


def process_unified_output(raw_output: str) -> Dict[str, Any]:
    """
    Process unified model output and extract results for all three tasks:
    binary classification, localization, and explanation.
    
    Args:
        raw_output: Raw text output from the model
        
    Returns:
        Dictionary containing results for all three evaluation types:
        {
            'binary': {'prediction': bool},
            'localization': [{'bbox_2d': [x1, y1, x2, y2]}, ...],
            'explanation': {'explanation': str}
        }
    """
    result = {
        'binary': {'prediction': False},
        'localization': [],
        'explanation': {'explanation': ''}
    }
    
    # Extract explanation text (full response for now, can be refined)
    result['explanation']['explanation'] = raw_output.strip()
    
    # Extract bounding boxes
    bboxes = extract_bboxes(raw_output)
    if bboxes:
        result['localization'] = [{"bbox_2d": bbox} for bbox in bboxes]
        result['binary']['prediction'] = True  # If bboxes found, artifacts exist
    
    # Look for explicit binary classification keywords
    lower_output = raw_output.lower()
    if "no artifact" in lower_output or "no defect" in lower_output or "false" in lower_output:
        result['binary']['prediction'] = False
    elif "true" in lower_output:
        result['binary']['prediction'] = True
    else:
        # If there are bboxes, there are artifacts
        if bboxes is None:
            result['binary']['prediction'] = False
        result['binary']['prediction'] = len(bboxes) > 0
    
    # Try to parse JSON responses if present
    try:
        # Look for JSON blocks
        json_match = re.search(r'```json\s*({.*?}|\[.*?\])\s*```', raw_output, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(1))
            
            # Handle different JSON formats
            if isinstance(parsed_json, dict):
                if 'prediction' in parsed_json:
                    result['binary']['prediction'] = parsed_json['prediction']
                if 'explanation' in parsed_json:
                    result['explanation']['explanation'] = parsed_json['explanation']
            elif isinstance(parsed_json, list) and len(parsed_json) > 0:
                # Assume it's a bbox list
                if all(isinstance(item, dict) and 'bbox_2d' in item for item in parsed_json):
                    result['localization'] = parsed_json
                    result['binary']['prediction'] = len(parsed_json) > 0
                    
    except (json.JSONDecodeError, KeyError):
        pass  # Keep existing parsing results
    
    return result


def process_finetuned_unified_output(raw_output: str) -> Dict[str, Any]:
    """
    Process finetuned model output and extract results for all three tasks.
    This is similar to process_unified_output but handles finetuned model specificities.
    
    Args:
        raw_output: Raw text output from the finetuned model
        
    Returns:
        Dictionary containing results for all three evaluation types
    """
    result = {
        'binary': {'prediction': False},
        'localization': [],
        'explanation': {'explanation': raw_output.strip()}
    }
    
    # Check for explicit false indicators
    if "false" in raw_output.lower():
        result['binary']['prediction'] = False
    elif "true" in raw_output.lower():
        result['binary']['prediction'] = True
    
    # Extract bounding boxes
    bboxes = extract_bboxes(raw_output)
    if bboxes:
        result['localization'] = [{"bbox_2d": bbox} for bbox in bboxes]
        result['binary']['prediction'] = True
    
    return result


def setup_logging(output_dir: str, dataset_type: str, model: str, use_finetuned: bool, finetune_path: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        dataset_type: Dataset name for log file naming
        model: Model name
        use_finetuned: Whether using finetuned model
        finetune_path: Finetune mode identifier
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(output_dir) / model
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if use_finetuned:
        log_mode_dir = log_dir / finetune_path
        log_mode_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_mode_dir / f'{timestamp}_{dataset_type}_finetuned.log'
    else:
        log_file = log_dir / f'{timestamp}_{dataset_type}.log'
    
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
        elif self.dataset_type == "ours":
            self._load_ours()
        elif self.dataset_type == "val":
            self._load_val_set()
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

        elif self.dataset_type == "ours":
            json_data = sample
            image_path = self.base_dir / f"images/{json_data['id']}.png"
            return json_data, image_path

        elif self.dataset_type == "val":
            json_data = sample
            image_path = sample['images'][0]
            json_path = os.path.join(os.path.dirname(image_path), "metadata.json")
            has_artifacts = True if "artifact_image" in image_path else False
            image_path = Path(image_path)
            if has_artifacts:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
            else:
                json_data = {}
                json_data['artifacts'] = []
                json_data['caption'] = "There are no artifacts in this image."
            json_data['has_artifacts'] = has_artifacts
            # print(json_data)
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
    
    def _load_ours(self):
        """Load custom eval dataset"""
        json_path = self.base_dir / "metadata.json"
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def _load_val_set(self):
        """Load the validation set used for training"""
        json_path = self.base_dir
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


def unified_inference(model, image: Image.Image, prompt: str, logger) -> Dict[str, Any]:
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
        
        logger.info(f"Raw response: {result}")

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


def unified_batch_inference(model, images: List[Image.Image], prompt: str, logger) -> List[Dict[str, Any]]:
    """
    Unified batch inference wrapper that handles different model signatures.
    
    Args:
        model: Model instance
        images: List of PIL Images to analyze  
        prompt: Text prompt for inference
        
    Returns:
        List of dictionaries containing standardized inference results
    """
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
        else:
            results = [model.inference(img, prompt) for img in images]
    
    # Standardize each result
    standardized_results = []
    for result in results:
        logger.info(f"Raw result: {result}")
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


def extract_all_task_results(unified_result: Dict[str, Any], use_finetuned: bool = False) -> Dict[str, Any]:
    """
    Extract results for all three tasks from unified inference result.
    
    Args:
        unified_result: Result from unified_inference
        use_finetuned: Whether to use finetuned model output processing
        
    Returns:
        Dictionary containing results for all three tasks:
        {
            'binary': {...},
            'localization': [...],
            'explanation': {...}
        }
    """
    # If there's an error, return empty results with error info
    # if "error" in unified_result:
    #     error_result = {
    #         "binary": {"error": unified_result["error"], "raw_response": unified_result.get("raw_response", "")},
    #         "localization": {"error": unified_result["error"], "raw_response": unified_result.get("raw_response", "")},
    #         "explanation": {"error": unified_result["error"], "raw_response": unified_result.get("raw_response", "")}
    #     }
    #     return error_result
    
    # If there's a heatmap (PAL/DiffDoctor models), handle specially
    # if "heatmap" in unified_result:
    #     return {
    #         "binary": unified_result,
    #         "localization": unified_result,
    #         "explanation": unified_result
    #     }
    
    # Extract raw response for processing
    raw_response = unified_result.get("raw_response", "")
    
    # Process based on whether using finetuned model
    if use_finetuned:
        all_results = process_finetuned_unified_output(raw_response)
    else:
        all_results = process_unified_output(raw_response)
    
    return all_results


def create_unified_prompt() -> str:
    """
    Create a unified prompt that asks for all three tasks simultaneously.
    This function provides a template that can be manually modified.
    
    Returns:
        Unified prompt string
    """
    # This is a placeholder - users should modify this manually
    # prompt = "Analyze the image and describe any visual anomalies. Provide bounding boxes and explain in detail. If there are no artifacts, simply return \"There are no artifacts in this image.\""
    prompt = "Analyze the image and describe any visual anomalies. Provide bounding boxes and explain in detail."

    return prompt


def run_unified_evaluation(config: Dict, max_samples: Optional[int] = None):
    """
    Run unified evaluation on dataset for all three tasks simultaneously.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    
    logger.info(f"Starting unified evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = create_model(config)
    data_iterator = DatasetIterator(config)
    
    # Initialize evaluators for all three task types
    binary_evaluator = Evaluation()
    loc_evaluator = Evaluation()
    exp_evaluator = Evaluation()
    
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

    # Create unified prompt (can be manually modified)
    prompt = create_unified_prompt()
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
            unified_output = unified_inference(model, image, prompt, logger)
            
            # Extract results for all three tasks
            all_task_results = extract_all_task_results(unified_output, config['use_finetuned'])
            
            # Evaluate each task separately
            binary_prediction = all_task_results['binary']
            loc_prediction = all_task_results['localization']
            exp_prediction = all_task_results['explanation']
            
            # Generate statistics for each task
            binary_stats = binary_evaluator.generate_statistics(
                dataset_type, 'binary', json_data, binary_prediction, image_size=image.size
            )
            
            loc_stats = loc_evaluator.generate_statistics(
                dataset_type, 'localization', json_data, loc_prediction, image_size=image.size
            )
            
            exp_stats = exp_evaluator.generate_statistics(
                dataset_type, 'explanation', json_data, exp_prediction, image_size=image.size
            )
            
            # For localization evaluation, also run LEGION and WSOL methods
            legion_stats = legion_evaluator.generate_statistics(
                dataset_type, 'localization', json_data, loc_prediction, image_size=image.size
            )
            wsol_stats = wsol_evaluator.generate_statistics(
                dataset_type, 'localization', json_data, loc_prediction, image_size=image.size
            )
            
            # Determine if sample has ground truth artifacts
            if dataset_type == 'synartifact':
                has_gt_artifacts = bool(json_data.get('Artifacts annotation', []))
            elif dataset_type == 'ours':
                has_gt_artifacts = json_data.get('has_artifacts', False)
            elif dataset_type == 'val':
                has_gt_artifacts = json_data.get('has_artifacts', False)
            else:
                has_gt_artifacts = True
            
            # Create comprehensive sample result
            sample_result = {
                'process_id': i + 1,
                'image_path': str(image_path),
                'has_gt_artifacts': has_gt_artifacts,
                
                # Binary classification results
                'binary_success': binary_stats['binary_success'],
                'classification': binary_stats['classification'],
                'has_pred_artifacts': binary_stats['has_pred_artifacts'],
                
                # Localization results (only for positive samples)
                'iou': loc_stats['iou'] if has_gt_artifacts else None,
                'loc_tp': loc_stats['loc_tp'] if has_gt_artifacts else None,
                'loc_fp': loc_stats['loc_fp'] if has_gt_artifacts else None,
                'loc_fn': loc_stats['loc_fn'] if has_gt_artifacts else None,
                'loc_precision': loc_stats['loc_precision'] if has_gt_artifacts else None,
                'loc_recall': loc_stats['loc_recall'] if has_gt_artifacts else None,
                'loc_f1': loc_stats['loc_f1'] if has_gt_artifacts else None,
                
                # LEGION evaluation metrics
                'legion_iou': legion_stats.get('iou') if has_gt_artifacts else None,
                'legion_miou': legion_stats.get('miou') if has_gt_artifacts else None,
                'legion_iou_foreground': legion_stats.get('iou_foreground') if has_gt_artifacts else None,
                'legion_iou_background': legion_stats.get('iou_background') if has_gt_artifacts else None,
                'legion_pixel_f1': legion_stats.get('pixel_f1') if has_gt_artifacts else None,
                'legion_pixel_precision': legion_stats.get('pixel_precision') if has_gt_artifacts else None,
                'legion_pixel_recall': legion_stats.get('pixel_recall') if has_gt_artifacts else None,
                
                # WSOL evaluation metrics
                'wsol_iou': wsol_stats.get('iou') if has_gt_artifacts else None,
                'artifact_type_stats': loc_stats.get('artifact_type_stats') if has_gt_artifacts and dataset_type == 'val' else None,
                
                # Explanation results
                'rouge_l': exp_stats['rouge_l'],
                'css': exp_stats['css'],
                
                # Store predictions for debugging
                'predictions': {
                    'binary': binary_prediction,
                    'localization': loc_prediction,
                    'explanation': exp_prediction
                }
            }
            
            # Log comprehensive results (use LEGION stats for intermediate logging)
            logger.info(f"Sample {i + 1} Results:")
            logger.info(f"  Binary: {sample_result['binary_success']} (Pred: {sample_result['has_pred_artifacts']}, GT: {sample_result['has_gt_artifacts']})")
            if has_gt_artifacts:
                legion_iou = legion_stats.get('iou', 0.0) if legion_stats.get('iou') is not None else 0.0
                legion_miou = legion_stats.get('miou', 0.0) if legion_stats.get('miou') is not None else 0.0
                logger.info(f"  LEGION: IoU={legion_iou:.3f}, mIoU={legion_miou:.3f}, Pixel F1={sample_result.get('legion_pixel_f1', 0.0):.3f}")
                logger.info(f"  WSOL: IoU={sample_result.get('wsol_iou', 0.0):.3f}")
                
                # Log artifact type matching for 'val' dataset
                if dataset_type == 'val' and sample_result.get('artifact_type_stats') is not None:
                    type_stats = sample_result.get('artifact_type_stats')
                    logger.info(f"    Artifact Types - Addition: {type_stats['addition']['matched']}/{type_stats['addition']['total']}, "
                              f"Removal: {type_stats['removal']['matched']}/{type_stats['removal']['total']}, "
                              f"Distortion: {type_stats['distortion']['matched']}/{type_stats['distortion']['total']}, "
                              f"Fusion: {type_stats['fusion']['matched']}/{type_stats['fusion']['total']}")
            logger.info(f"  Explanation: ROUGE-L={sample_result['rouge_l']:.3f}, CSS={sample_result['css']:.3f}")

            # Store results
            results[i] = sample_result
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            continue

    logger.info("Unified evaluation completed!")
    
    # Compute comprehensive summary statistics
    if results:
        total_samples = len(results)
        
        # Binary classification metrics
        binary_accuracy = sum(r.get('binary_success', False) for _, r in results.items()) / total_samples
        f1_metrics = binary_evaluator.compute_f1_metrics(results)
        
        # Localization metrics (only for positive samples)
        valid_loc_results = [r for _, r in results.items() if r.get('has_gt_artifacts') is True]
        if valid_loc_results:
            mean_iou = sum(r.get('iou', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            mean_loc_f1 = sum(r.get('loc_f1', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            mean_loc_precision = sum(r.get('loc_precision', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            mean_loc_recall = sum(r.get('loc_recall', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            total_loc_tp = sum(r.get('loc_tp', 0) for r in valid_loc_results if r.get('loc_tp') is not None)
            total_loc_fp = sum(r.get('loc_fp', 0) for r in valid_loc_results if r.get('loc_fp') is not None)
            total_loc_fn = sum(r.get('loc_fn', 0) for r in valid_loc_results if r.get('loc_fn') is not None)
            
            # LEGION metrics
            legion_valid_results = [r for r in valid_loc_results if r.get('legion_iou') is not None]
            legion_mean_iou = sum(r.get('legion_iou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_miou = sum(r.get('legion_miou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_fg = sum(r.get('legion_iou_foreground', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_bg = sum(r.get('legion_iou_background', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_f1 = sum(r.get('legion_pixel_f1', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_precision = sum(r.get('legion_pixel_precision', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_recall = sum(r.get('legion_pixel_recall', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            
            # WSOL metrics
            wsol_valid_results = [r for r in valid_loc_results if r.get('wsol_iou') is not None]
            wsol_mean_iou = sum(r.get('wsol_iou', 0.0) for r in wsol_valid_results) / len(wsol_valid_results) if wsol_valid_results else 0.0
            
            # Aggregate artifact type statistics for 'val' dataset
            artifact_type_stats = None
            if dataset_type == "val":
                artifact_type_stats = loc_evaluator.aggregate_artifact_type_stats(results)
        else:
            mean_iou = mean_loc_f1 = mean_loc_precision = mean_loc_recall = 0.0
            total_loc_tp = total_loc_fp = total_loc_fn = 0
            legion_mean_iou = legion_mean_miou = legion_mean_iou_fg = legion_mean_iou_bg = 0.0
            legion_mean_pixel_f1 = legion_mean_pixel_precision = legion_mean_pixel_recall = 0.0
            wsol_mean_iou = 0.0
        
        # Explanation metrics
        mean_rouge_l = sum(r.get('rouge_l', 0.0) for _, r in results.items()) / total_samples
        mean_css = sum(r.get('css', 0.0) for _, r in results.items()) / total_samples
        
        # Comprehensive reporting
        logger.info("=" * 100)
        logger.info("UNIFIED EVALUATION SUMMARY - ALL TASKS")
        logger.info("=" * 100)
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Valid localization samples (positive): {len(valid_loc_results) if valid_loc_results else 0}")
        logger.info("")
        
        # Binary classification results
        logger.info("🎯 BINARY CLASSIFICATION RESULTS:")
        logger.info(f"  Accuracy: {binary_accuracy:.3f}")
        logger.info(f"  TP: {f1_metrics.get('tp', 0)}, FP: {f1_metrics.get('fp', 0)}, FN: {f1_metrics.get('fn', 0)}, TN: {f1_metrics.get('tn', 0)}")
        logger.info(f"  Precision: {f1_metrics.get('precision', 0.0):.3f}")
        logger.info(f"  Recall: {f1_metrics.get('recall', 0.0):.3f}")
        logger.info(f"  F1-Score: {f1_metrics.get('f1_positive', 0.0):.3f}")
        logger.info(f"  Macro F1: {f1_metrics.get('macro_f1', 0.0):.3f}")
        logger.info("")
        
        # Localization results
        if valid_loc_results:
            logger.info("📍 LOCALIZATION RESULTS:")
            logger.info("  📊 STANDARD EVALUATION (Threshold-Independent Bbox Metrics):")
            logger.info(f"    Mean IoU: {mean_iou:.3f}")
            logger.info(f"    Mean F1: {mean_loc_f1:.3f}")
            logger.info(f"    Mean Precision: {mean_loc_precision:.3f}")
            logger.info(f"    Mean Recall: {mean_loc_recall:.3f}")
            logger.info(f"    Total TP/FP/FN: {total_loc_tp}/{total_loc_fp}/{total_loc_fn}")
            
            # Global F1
            global_precision = total_loc_tp / (total_loc_tp + total_loc_fp) if (total_loc_tp + total_loc_fp) > 0 else 0.0
            global_recall = total_loc_tp / (total_loc_tp + total_loc_fn) if (total_loc_tp + total_loc_fn) > 0 else 0.0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
            logger.info(f"    Global Precision: {global_precision:.3f}")
            logger.info(f"    Global Recall: {global_recall:.3f}")
            logger.info(f"    Global F1: {global_f1:.3f}")
            
            # Artifact type statistics for 'val' dataset
            if dataset_type == "val" and artifact_type_stats is not None:
                logger.info("")
                logger.info("    📋 PER-ARTIFACT-TYPE STATISTICS:")
                for artifact_type in ['addition', 'removal', 'distortion', 'fusion']:
                    stats = artifact_type_stats[artifact_type]
                    logger.info(f"      {artifact_type.capitalize()}: {stats['matched']}/{stats['total']} detected (rate: {stats['detection_rate']:.3f})")
            logger.info("")
            
            # LEGION evaluation
            logger.info("  🎯 LEGION EVALUATION (Pixel-Level Segmentation Metrics):")
            logger.info(f"    Valid samples: {len(legion_valid_results)}")
            logger.info(f"    Mean IoU (Legacy): {legion_mean_iou:.3f}")
            logger.info(f"    Mean IoU (mIoU): {legion_mean_miou:.3f}")
            logger.info(f"      - Foreground IoU: {legion_mean_iou_fg:.3f}")
            logger.info(f"      - Background IoU: {legion_mean_iou_bg:.3f}")
            logger.info(f"    Pixel F1 Score: {legion_mean_pixel_f1:.3f}")
            logger.info(f"    Pixel Precision: {legion_mean_pixel_precision:.3f}")
            logger.info(f"    Pixel Recall: {legion_mean_pixel_recall:.3f}")
            logger.info("")
            
            # WSOL evaluation
            logger.info("  🔄 WSOL EVALUATION (Threshold-Independent IoU):")
            logger.info(f"    Valid samples: {len(wsol_valid_results)}")
            logger.info(f"    Mean IoU: {wsol_mean_iou:.3f}")
            logger.info("")
        else:
            logger.info("📍 LOCALIZATION RESULTS: No positive samples to evaluate")
            logger.info("")
        
        # Explanation results
        logger.info("📝 EXPLANATION RESULTS:")
        logger.info(f"  Mean ROUGE-L: {mean_rouge_l:.3f}")
        logger.info(f"  Mean CSS: {mean_css:.3f}")
        logger.info("")

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


def run_unified_batch_evaluation(config: Dict, max_samples: Optional[int] = None):
    """
    Run unified batch evaluation on dataset for all three tasks simultaneously.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    
    logger.info(f"Starting unified batch evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = create_model(config)
    data_iterator = DatasetIterator(config)
    
    # Initialize evaluators for all three task types
    binary_evaluator = Evaluation()
    loc_evaluator = Evaluation()
    exp_evaluator = Evaluation()
    
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

    prompt = create_unified_prompt()
    logger.info(f"Input query: {prompt}")

    # Determine batch size: default to 2 if not provided in config
    target_batch_size: int = int(config.get('batch_size', 2) or 2)
    current_batch_size: int = target_batch_size

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
                if dataset_type == 'richhf':
                    image = image.resize((512, 512), Image.LANCZOS)
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
            batch_unified_results = unified_batch_inference(model, batch_images, prompt, logger)
            batch_all_task_results = [extract_all_task_results(result, config['use_finetuned']) for result in batch_unified_results]
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and len(batch_images) > 1:
                logger.warning("OOM during batched inference. Falling back to per-sample inference for this batch.")
                # Reduce future batch size to be more conservative
                current_batch_size = max(1, current_batch_size // 2)
                batch_all_task_results = []
                for img in batch_images:
                    try:
                        unified_result = unified_inference(model, img, prompt, logger)
                        batch_all_task_results.append(extract_all_task_results(unified_result, config['use_finetuned']))
                    except Exception as inner_e:
                        logger.error(f"Per-sample inference failed: {inner_e}")
                        batch_all_task_results.append({
                            'binary': {'error': str(inner_e)},
                            'localization': {'error': str(inner_e)},
                            'explanation': {'error': str(inner_e)}
                        })
            else:
                raise
        
        # Evaluate each item in the batch
        for idx, (json_data, image_path, image, all_task_results) in enumerate(
            zip(batch_json_data, batch_image_paths, batch_images, batch_all_task_results)
        ):
            # Extract results for each task
            binary_prediction = all_task_results['binary']
            loc_prediction = all_task_results['localization']
            exp_prediction = all_task_results['explanation']

            # Generate statistics for each task
            binary_stats = binary_evaluator.generate_statistics(
                dataset_type, 'binary', json_data, binary_prediction, image_size=image.size
            )
            
            loc_stats = loc_evaluator.generate_statistics(
                dataset_type, 'localization', json_data, loc_prediction, image_size=image.size
            )
            
            exp_stats = exp_evaluator.generate_statistics(
                dataset_type, 'explanation', json_data, exp_prediction, image_size=image.size
            )
            
            # For localization evaluation, also run LEGION and WSOL methods
            legion_stats = legion_evaluator.generate_statistics(
                dataset_type, 'localization', json_data, loc_prediction, image_size=image.size
            )
            wsol_stats = wsol_evaluator.generate_statistics(
                dataset_type, 'localization', json_data, loc_prediction, image_size=image.size
            )
            
            # Determine if sample has ground truth artifacts
            if dataset_type == 'synartifact':
                has_gt_artifacts = bool(json_data.get('Artifacts annotation', []))
            elif dataset_type == 'ours':
                has_gt_artifacts = json_data.get('has_artifacts', False)
            elif dataset_type == 'val':
                has_gt_artifacts = json_data.get('has_artifacts', False)
            else:
                has_gt_artifacts = True
            
            # Create comprehensive sample result
            sample_result = {
                'process_id': processed + idx + 1,
                'image_path': str(image_path),
                'has_gt_artifacts': has_gt_artifacts,
                
                # Binary classification results
                'binary_success': binary_stats['binary_success'],
                'classification': binary_stats['classification'],
                'has_pred_artifacts': binary_stats['has_pred_artifacts'],
                
                # Localization results (only for positive samples)
                'iou': loc_stats['iou'] if has_gt_artifacts else None,
                'loc_tp': loc_stats['loc_tp'] if has_gt_artifacts else None,
                'loc_fp': loc_stats['loc_fp'] if has_gt_artifacts else None,
                'loc_fn': loc_stats['loc_fn'] if has_gt_artifacts else None,
                'loc_precision': loc_stats['loc_precision'] if has_gt_artifacts else None,
                'loc_recall': loc_stats['loc_recall'] if has_gt_artifacts else None,
                'loc_f1': loc_stats['loc_f1'] if has_gt_artifacts else None,
                'artifact_type_stats': loc_stats.get('artifact_type_stats') if has_gt_artifacts and dataset_type == 'val' else None,
                
                # LEGION evaluation metrics
                'legion_iou': legion_stats.get('iou') if has_gt_artifacts else None,
                'legion_miou': legion_stats.get('miou') if has_gt_artifacts else None,
                'legion_iou_foreground': legion_stats.get('iou_foreground') if has_gt_artifacts else None,
                'legion_iou_background': legion_stats.get('iou_background') if has_gt_artifacts else None,
                'legion_pixel_f1': legion_stats.get('pixel_f1') if has_gt_artifacts else None,
                'legion_pixel_precision': legion_stats.get('pixel_precision') if has_gt_artifacts else None,
                'legion_pixel_recall': legion_stats.get('pixel_recall') if has_gt_artifacts else None,
                
                # WSOL evaluation metrics
                'wsol_iou': wsol_stats.get('iou') if has_gt_artifacts else None,
                
                
                # Explanation results
                'rouge_l': exp_stats['rouge_l'],
                'css': exp_stats['css'],
                
                # Store predictions for debugging
                'predictions': {
                    'binary': binary_prediction,
                    'localization': loc_prediction,
                    'explanation': exp_prediction
                }
            }
            
            # Log comprehensive results (use LEGION stats for intermediate logging)
            logger.info(f"Sample {processed + idx + 1} Results:")
            logger.info(f"  Binary: {sample_result['binary_success']} (Pred: {sample_result['has_pred_artifacts']}, GT: {sample_result['has_gt_artifacts']})")
            if has_gt_artifacts:
                legion_iou = legion_stats.get('iou', 0.0) if legion_stats.get('iou') is not None else 0.0
                legion_miou = legion_stats.get('miou', 0.0) if legion_stats.get('miou') is not None else 0.0
                logger.info(f"  LEGION: IoU={legion_iou:.3f}, mIoU={legion_miou:.3f}, Pixel F1={sample_result.get('legion_pixel_f1', 0.0):.3f}")
                logger.info(f"  WSOL: IoU={sample_result.get('wsol_iou', 0.0):.3f}")
                
                # Log artifact type matching for 'val' dataset
                if dataset_type == 'val' and sample_result.get('artifact_type_stats') is not None:
                    type_stats = sample_result.get('artifact_type_stats')
                    logger.info(f"    Artifact Types - Addition: {type_stats['addition']['matched']}/{type_stats['addition']['total']}, "
                              f"Removal: {type_stats['removal']['matched']}/{type_stats['removal']['total']}, "
                              f"Distortion: {type_stats['distortion']['matched']}/{type_stats['distortion']['total']}, "
                              f"Fusion: {type_stats['fusion']['matched']}/{type_stats['fusion']['total']}")
            logger.info(f"  Explanation: ROUGE-L={sample_result['rouge_l']:.3f}, CSS={sample_result['css']:.3f}")

            results[processed + idx] = sample_result

        processed += len(batch_images)

        if config.get('model_type') == 'pal' and isinstance(model, PalEval) and processed % (10 * current_batch_size) == 0:
            model.clear_gpu_cache()
            logger.info(f"🧹 Cleared GPU cache after processing {processed} samples")
    
    logger.info("Unified batch evaluation completed!")
    
    # Compute comprehensive summary statistics
    if results:
        total_samples = len(results)
        
        # Binary classification metrics
        binary_accuracy = sum(r.get('binary_success', False) for _, r in results.items()) / total_samples
        f1_metrics = binary_evaluator.compute_f1_metrics(results)
        
        # Localization metrics (only for positive samples)
        valid_loc_results = [r for _, r in results.items() if r.get('has_gt_artifacts') is True]
        artifact_type_stats = None
        if valid_loc_results:
            mean_iou = sum(r.get('iou', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            mean_loc_f1 = sum(r.get('loc_f1', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            mean_loc_precision = sum(r.get('loc_precision', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            mean_loc_recall = sum(r.get('loc_recall', 0.0) for r in valid_loc_results) / len(valid_loc_results)
            total_loc_tp = sum(r.get('loc_tp', 0) for r in valid_loc_results if r.get('loc_tp') is not None)
            total_loc_fp = sum(r.get('loc_fp', 0) for r in valid_loc_results if r.get('loc_fp') is not None)
            total_loc_fn = sum(r.get('loc_fn', 0) for r in valid_loc_results if r.get('loc_fn') is not None)
            
            # LEGION metrics
            legion_valid_results = [r for r in valid_loc_results if r.get('legion_iou') is not None]
            legion_mean_iou = sum(r.get('legion_iou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_miou = sum(r.get('legion_miou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_fg = sum(r.get('legion_iou_foreground', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_bg = sum(r.get('legion_iou_background', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_f1 = sum(r.get('legion_pixel_f1', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_precision = sum(r.get('legion_pixel_precision', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_recall = sum(r.get('legion_pixel_recall', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            
            # WSOL metrics
            wsol_valid_results = [r for r in valid_loc_results if r.get('wsol_iou') is not None]
            wsol_mean_iou = sum(r.get('wsol_iou', 0.0) for r in wsol_valid_results) / len(wsol_valid_results) if wsol_valid_results else 0.0

            if dataset_type == "val":
                artifact_type_stats = loc_evaluator.aggregate_artifact_type_stats(results)

        else:
            mean_iou = mean_loc_f1 = mean_loc_precision = mean_loc_recall = 0.0
            total_loc_tp = total_loc_fp = total_loc_fn = 0
            legion_mean_iou = legion_mean_miou = legion_mean_iou_fg = legion_mean_iou_bg = 0.0
            legion_mean_pixel_f1 = legion_mean_pixel_precision = legion_mean_pixel_recall = 0.0
            wsol_mean_iou = 0.0
        
        # Explanation metrics
        mean_rouge_l = sum(r.get('rouge_l', 0.0) for _, r in results.items()) / total_samples
        mean_css = sum(r.get('css', 0.0) for _, r in results.items()) / total_samples
        
        # Comprehensive reporting
        logger.info("=" * 100)
        logger.info("UNIFIED BATCH EVALUATION SUMMARY - ALL TASKS")
        logger.info("=" * 100)
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Valid localization samples (positive): {len(valid_loc_results) if valid_loc_results else 0}")
        logger.info("")
        
        # Binary classification results
        logger.info("🎯 BINARY CLASSIFICATION RESULTS:")
        logger.info(f"  Accuracy: {binary_accuracy:.3f}")
        logger.info(f"  TP: {f1_metrics.get('tp', 0)}, FP: {f1_metrics.get('fp', 0)}, FN: {f1_metrics.get('fn', 0)}, TN: {f1_metrics.get('tn', 0)}")
        logger.info(f"  Precision: {f1_metrics.get('precision', 0.0):.3f}")
        logger.info(f"  Recall: {f1_metrics.get('recall', 0.0):.3f}")
        logger.info(f"  F1-Score: {f1_metrics.get('f1_positive', 0.0):.3f}")
        logger.info(f"  Macro F1: {f1_metrics.get('macro_f1', 0.0):.3f}")
        logger.info("")
        
        # Localization results
        if valid_loc_results:
            logger.info("📍 LOCALIZATION RESULTS:")
            logger.info("  📊 STANDARD EVALUATION (Threshold-Independent Bbox Metrics):")
            logger.info(f"    Mean IoU: {mean_iou:.3f}")
            logger.info(f"    Mean F1: {mean_loc_f1:.3f}")
            logger.info(f"    Mean Precision: {mean_loc_precision:.3f}")
            logger.info(f"    Mean Recall: {mean_loc_recall:.3f}")
            logger.info(f"    Total TP/FP/FN: {total_loc_tp}/{total_loc_fp}/{total_loc_fn}")
            
            # Global F1
            global_precision = total_loc_tp / (total_loc_tp + total_loc_fp) if (total_loc_tp + total_loc_fp) > 0 else 0.0
            global_recall = total_loc_tp / (total_loc_tp + total_loc_fn) if (total_loc_tp + total_loc_fn) > 0 else 0.0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
            logger.info(f"    Global Precision: {global_precision:.3f}")
            logger.info(f"    Global Recall: {global_recall:.3f}")
            logger.info(f"    Global F1: {global_f1:.3f}")
            
            # Artifact type statistics for 'val' dataset
            if dataset_type == "val" and artifact_type_stats is not None:
                logger.info("")
                logger.info("    📋 PER-ARTIFACT-TYPE STATISTICS:")
                for artifact_type in ['addition', 'removal', 'distortion', 'fusion']:
                    stats = artifact_type_stats[artifact_type]
                    logger.info(f"      {artifact_type.capitalize()}: {stats['matched']}/{stats['total']} detected (rate: {stats['detection_rate']:.3f})")
            logger.info("")
            
            # LEGION evaluation
            logger.info("  🎯 LEGION EVALUATION (Pixel-Level Segmentation Metrics):")
            logger.info(f"    Valid samples: {len(legion_valid_results)}")
            logger.info(f"    Mean IoU (Legacy): {legion_mean_iou:.3f}")
            logger.info(f"    Mean IoU (mIoU): {legion_mean_miou:.3f}")
            logger.info(f"      - Foreground IoU: {legion_mean_iou_fg:.3f}")
            logger.info(f"      - Background IoU: {legion_mean_iou_bg:.3f}")
            logger.info(f"    Pixel F1 Score: {legion_mean_pixel_f1:.3f}")
            logger.info(f"    Pixel Precision: {legion_mean_pixel_precision:.3f}")
            logger.info(f"    Pixel Recall: {legion_mean_pixel_recall:.3f}")
            logger.info("")
            
            # WSOL evaluation
            logger.info("  🔄 WSOL EVALUATION (Threshold-Independent IoU):")
            logger.info(f"    Valid samples: {len(wsol_valid_results)}")
            logger.info(f"    Mean IoU: {wsol_mean_iou:.3f}")
            logger.info("")
        else:
            logger.info("📍 LOCALIZATION RESULTS: No positive samples to evaluate")
            logger.info("")
        
        # Explanation results
        logger.info("📝 EXPLANATION RESULTS:")
        logger.info(f"  Mean ROUGE-L: {mean_rouge_l:.3f}")
        logger.info(f"  Mean CSS: {mean_css:.3f}")
        logger.info("")

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
    """Main function for unified model evaluation."""
    parser = argparse.ArgumentParser(
        description='Unified evaluation of VLM/MLLM models on all artifact detection tasks'
    )
    parser.add_argument('--model', type=str, choices=['qwen', 'intern', 'gpt', 'gemini', 'pal', 'diff', 'legion'], 
                       default='qwen', help='Model type to evaluate (default: qwen)')
    parser.add_argument('--dataset', type=str, 
                       choices=['synthscars', 'synartifact', 'loki', 'richhf', 'ours', 'val'], 
                       default='ours', help='Dataset to evaluate on (default: ours)')
    parser.add_argument('--use-finetuned', action='store_true',
                       help='Use finetuned model instead of base model')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device for inference (default: cuda:0)')
    parser.add_argument('--log-dir', type=str, default='eval_all_logs',
                       help='Directory for logs (default: eval_all_logs)')
    parser.add_argument('--output-dir', type=str, default='eval_all_results',
                       help='Directory for results (default: eval_all_results)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Custom base directory for dataset')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1, use >1 for batch processing)')
    parser.add_argument('--use-multi-gpu', action='store_true',
                       help='Enable multi-GPU inference for PAL model')
    parser.add_argument('--gpu-devices', type=str, nargs='+', default=None,
                       help='Specify GPU devices to use (e.g., 0 1 or cuda:0 cuda:1)')
    parser.add_argument('--finetune-path', type=str)
                       
    args = parser.parse_args()
    
    # Set dataset paths if not provided
    if args.base_dir is None:
        dataset_paths = {
            'synthscars': "/home/jovyan/image-artifacts/data/SynthScars/test",
            'synartifact': "/home/jovyan/image-artifacts/data/SynArtifact/data",
            'loki': "/home/jovyan/image-artifacts/data/loki",
            'richhf': "/home/jovyan/image-artifacts/data/richhf-18k",
            'ours': "/home/jovyan/image-artifacts/data/eval",
            'val': "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/data/artifact_1k.json"
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
        'base_dir': base_dir,
        'log_dir': args.log_dir,
        'use_finetuned': args.use_finetuned,
        'device': args.device,
        'batch_size': args.batch_size,
        'use_multi_gpu': args.use_multi_gpu,
        'gpu_devices': args.gpu_devices,
        'finetune_path': args.finetune_path
    }
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.dataset, args.model, args.use_finetuned, args.finetune_path)
    
    logger.info(f"🚀 Starting unified evaluation for {args.dataset.upper()} dataset")
    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"🤖 Finetuned: {args.use_finetuned}")
    logger.info(f"🗒️ Evaluating: ALL TASKS (binary + localization + explanation)")
    logger.info(f"📁 Dataset path: {base_dir}")
    logger.info(f"🔧 Device: {args.device}")
    logger.info(f"📦 Batch size: {args.batch_size}")

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
        # Run unified evaluation (choose between batch and single mode)
        if args.batch_size > 1:
            logger.info("🔄 Running in batch mode")
            results = run_unified_batch_evaluation(config, args.max_samples)
        else:
            logger.info("🔄 Running in single-sample mode")
            results = run_unified_evaluation(config, args.max_samples)
        
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        if args.use_finetuned:
            output_dir = Path(args.output_dir)
            output_specified_dir = output_dir / args.model / args.finetune_path
            output_specified_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_specified_dir / f"{timestamp}_results_{args.dataset}.json"
        else:
            output_dir = Path(args.output_dir)
            output_specified_dir = output_dir / args.model
            output_specified_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_specified_dir / f"{timestamp}_results_{args.dataset}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Unified evaluation completed! Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
