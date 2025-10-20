"""
Multi-Task VQA Evaluation for Artifact Detection Models

This evaluation script tests three independent single-image tasks:
  - Task 1.1: Binary Detection (single image)
  - Task 1.2: Localization (single image)  
  - Task 1.3: Global Explanation (single image)

Note: This does NOT evaluate:
  - Task 1.4: Regional Explanation (evaluated implicitly via other metrics)
  - Tasks 4.1-4.4: Pair-image tasks (requires two images)
  
Supports batch inference for improved performance.
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

from models import QwenEval, InternEval, GPTEval, GeminiEval, PalEval, DiffEval, LegionEval
from eval_utils import Evaluation
import legion_eval_utils
import wsol_eval_utils

# Add path for VQA prompts
sys.path.append(str(Path(__file__).parent.parent / "train" / "LLaMA-Factory" / "data_gen"))
from vqa_gen.vqa_prompts import VQAPrompts

logger = logging.getLogger(__name__)


def extract_bboxes(text: str) -> List[List[int]]:
    """
    Extracts all 4-number bounding boxes that appear in the form:
        [x1, y1, x2, y2]: <optional description>
    Returns a list of [x1, y1, x2, y2]. If none are found, returns [].
    """
    pattern = re.compile(
        r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*:',
        flags=re.UNICODE
    )

    bboxes = []
    for x1, y1, x2, y2 in pattern.findall(text):
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    return bboxes


def parse_binary_detection_response(response: str) -> Dict[str, Any]:
    """Parse binary detection JSON response.
    
    Expected format: {"type":"binary_detection","artifact_present":"yes|no"}
    
    Fallback to text-based format if JSON parsing fails.
    
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
        # Fallback to text-based detection using bbox extraction
        bboxes = extract_bboxes(response)
        if bboxes:
            return {'prediction': True}
        # Additional fallback to text-based detection
        if "yes" in response.lower() and "no" not in response.lower():
            return {'prediction': True}
        return {'prediction': False}


def parse_localization_response(response: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Parse localization JSON response.
    
    Expected format: {"type":"localization","coord_space":"pixel","bboxes":[{"bbox":[xmin,ymin,xmax,ymax]}, ...]}
    
    Fallback to text-based format if JSON parsing fails.
    
    Returns:
        List of dicts with 'bbox_2d' field
    """
    if not response or not response.strip():
        return []
    
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
        logger.warning(f"Failed to parse localization JSON: {e}, response: {response[:100]}...")
        
        # Fallback: Try text-based bbox extraction
        bboxes = extract_bboxes(response)
        bbox_list = []
        for bbox in bboxes:
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Validate and clamp coordinates
                x1 = max(0, min(image_width, int(x1)))
                y1 = max(0, min(image_height, int(y1)))
                x2 = max(0, min(image_width, int(x2)))
                y2 = max(0, min(image_height, int(y2)))
                
                # Ensure valid bbox (x2 > x1, y2 > y1)
                if x2 > x1 and y2 > y1:
                    bbox_list.append({"bbox_2d": [x1, y1, x2, y2]})
        
        # Fallback 2: Try to extract coordinate arrays from malformed responses
        if not bbox_list:
            coord_patterns = [
                r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',  # [x1,y1,x2,y2]
                r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)',      # x1,y1,x2,y2
            ]
            
            for pattern in coord_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    try:
                        x1, y1, x2, y2 = map(int, match)
                        # Validate and clamp coordinates
                        x1 = max(0, min(image_width, x1))
                        y1 = max(0, min(image_height, y1))
                        x2 = max(0, min(image_width, x2))
                        y2 = max(0, min(image_height, y2))
                        
                        # Ensure valid bbox (x2 > x1, y2 > y1)
                        if x2 > x1 and y2 > y1:
                            bbox_list.append({"bbox_2d": [x1, y1, x2, y2]})
                    except (ValueError, TypeError) as parse_error:
                        logger.warning(f"Failed to parse coordinates {match}: {parse_error}")
                        continue
        
        if bbox_list:
            logger.info(f"Successfully extracted {len(bbox_list)} bboxes from malformed response")
        
        return bbox_list


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


def calculate_f1_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Calculate F1, precision, and recall from confusion matrix components.
    
    Args:
        tp: True positives
        fp: False positives  
        tn: True negatives
        fn: False negatives
        
    Returns:
        Dict with 'precision', 'recall', 'f1' scores
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def setup_logging(output_dir: str, dataset_type: str, model_type: str, exp_name: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        dataset_type: Dataset name for log file naming
        model_type: Model type for log file naming
        exp_name: Experiment name for log file naming
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'multi_task_vqa_{dataset_type}_{model_type}_{exp_name}_{timestamp}.log'
    
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
    
    Supports SynthScars, SynArtifact, LOKI, RichHF, and custom datasets.
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
        
        # For val and train datasets, use dataset_path directly
        if self.dataset_type in ["val", "train"]:
            self.dataset_path = config.get('dataset_path')
            if not self.dataset_path:
                raise ValueError(f"dataset_path is required for {self.dataset_type} dataset")
        
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
        elif self.dataset_type == "train":
            self._load_train_set()
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
            image_rel = sample.lstrip('./')
            if image_rel.startswith('images/'):
                image_path = self.base_dir / image_rel
            else:
                image_path = self.base_dir / 'images' / image_rel
            image_id = Path(image_path).stem
            json_path = self.base_dir / 'data' / 'annotation_json_artifacts_class' / f"{image_id}.json"
            
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

        elif self.dataset_type in ["val", "train"]:
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
            return json_data, image_path
        
        raise RuntimeError("Unsupported dataset type in _process_sample")
    
    def _load_synthscars(self):
        """Load SynthScars dataset."""
        json_path = self.base_dir / "annotations" / "test.json"
        with open(json_path, "rb") as f:
            self.data = json.load(f)
    
    def _load_synartifact(self):
        """Load SynArtifact dataset."""
        eval_set_primary = self.base_dir / "eval.txt"
        eval_set_alt = self.base_dir / "data" / "eval.txt"
        self.data = []
        eval_file = None
        if eval_set_primary.exists():
            eval_file = eval_set_primary
        elif eval_set_alt.exists():
            eval_file = eval_set_alt
        else:
            # Fallback: enumerate images directory
            images_dir = self.base_dir / "images"
            if images_dir.exists():
                for p in images_dir.rglob("*.png"):
                    rel = p.relative_to(images_dir)
                    self.data.append(str(rel).replace("\\", "/"))
            return
        with open(eval_file, "r") as f:
            for line in f:
                item = line.strip()
                if item:
                    self.data.append(item)
    
    def _load_loki(self):
        """Load LOKI dataset."""
        json_path = self.base_dir / "open_ended_vqa.json"
        with open(json_path, "rb") as f:
            self.data = json.load(f)

    def _load_richhf(self):
        """Load RichHF-18K dataset from TFRecord file."""
        json_path = os.path.join(self.base_dir, "test.json")
        with open(json_path, "r") as f:
            loaded = json.load(f)
            # Ensure iterable by index: convert dicts to list of values
            if isinstance(loaded, dict):
                self.data = list(loaded.values())
            else:
                self.data = loaded
    
    def _load_ours(self):
        """Load custom eval dataset"""
        json_path = self.base_dir / "metadata.json"
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def _load_val_set(self):
        """Load the validation set used for training"""
        with open(self.dataset_path, "r") as f:
            self.data = json.load(f)
    
    def _load_train_set(self):
        """Load the training set used for training"""
        with open(self.dataset_path, "r") as f:
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


def unified_inference(model, image: Image.Image, prompt: str) -> str:
    """
    Unified inference wrapper for different model types.
    
    Args:
        model: Model instance 
        image: PIL Image to analyze
        prompt: Text prompt for inference
        
    Returns:
        String containing the model's response
    """
    # Handle models that only take image (no prompt)
    if isinstance(model, (PalEval, DiffEval, LegionEval)):
        result = model.inference(image)
    else:
        # Models that take both image and prompt
        result = model.inference(image, prompt)
    
    # Handle None returns
    if result is None:
        return ""
    
    # Handle string returns
    if isinstance(result, str):
        return result
    
    # Handle dictionary returns
    if isinstance(result, dict):
        # Try to extract raw_response
        if "raw_response" in result:
            return result["raw_response"]
        # Try to extract parsed_output as string
        elif "parsed_output" in result:
            return str(result["parsed_output"])
        else:
            return str(result)
    
    # Handle other types
    return str(result)


def unified_batch_inference(model, images: List[Image.Image], prompt: str) -> List[str]:
    """
    Unified batch inference wrapper for different model types.
    
    Args:
        model: Model instance
        images: List of PIL Images to analyze  
        prompt: Text prompt for inference
        
    Returns:
        List of strings containing the model's responses
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
    
    # Standardize each result to string
    standardized_results = []
    for result in results:
        if result is None:
            standardized_results.append("")
        elif isinstance(result, str):
            standardized_results.append(result)
        elif isinstance(result, dict):
            if "raw_response" in result:
                standardized_results.append(result["raw_response"])
            elif "parsed_output" in result:
                standardized_results.append(str(result["parsed_output"]))
            else:
                standardized_results.append(str(result))
        else:
            standardized_results.append(str(result))
    
    return standardized_results


def get_prompts() -> Dict[str, str]:
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


def process_sample(args, gt, image_path, image, binary_output, loc_output, expl_output,
                   evaluator, legion_evaluator, wsol_evaluator, total_processed):
    """Process a single sample and return results."""
    
    # Parse outputs for each task
    binary_pred = parse_binary_detection_response(binary_output)
    loc_pred = parse_localization_response(loc_output, image.size[0], image.size[1])
    expl_pred = parse_global_explanation_response(expl_output)
    
    # Determine if GT has artifacts
    if args.dataset in ['ours', 'val', 'train']:
        has_gt = gt.get('has_artifacts', False)
    elif args.dataset == 't2i':
        has_gt = bool(gt.get('Artifacts annotation', []))
    elif args.dataset == 'synartifact':
        has_gt = bool(gt.get('Artifacts annotation', []))
    else:
        has_gt = True
    
    # Pre-process GT to add fields expected by eval_utils
    if args.dataset in ['ours', 'val', 'train']:
        if 'bboxes' not in gt:
            gt_artifacts = gt.get('artifacts', [])
            gt['bboxes'] = [
                artifact.get('target_bbox', artifact.get('bbox', artifact.get('bbox_2d', [])))
                for artifact in gt_artifacts
            ]
        
        if 'explanation' not in gt and 'caption' in gt:
            gt['explanation'] = gt['caption']
    
    elif args.dataset == 'synartifact':
        gt_artifacts = gt.get('Artifacts annotation', [])
        gt['bboxes'] = []
        for artifact in gt_artifacts:
            rect_start = artifact.get('rect_start', [])
            rect_end = artifact.get('rect_end', [])
            if len(rect_start) == 2 and len(rect_end) == 2:
                x1, y1 = rect_start
                x2, y2 = rect_end
                gt['bboxes'].append([x1, y1, x2, y2])
        
        if gt_artifacts:
            explanations = [a.get('artifacts_caption', '') for a in gt_artifacts if a.get('artifacts_caption')]
            gt['explanation'] = '; '.join(explanations) if explanations else 'No artifacts'
        else:
            gt['explanation'] = 'No artifacts'
    
    # Convert prediction format for eval_utils
    loc_pred_formatted = []
    img_w, img_h = image.size
    for detection in loc_pred:
        if 'bbox_2d' in detection and detection['bbox_2d']:
            bbox = detection['bbox_2d']
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(img_w-1, int(x1)))
                y1 = max(0, min(img_h-1, int(y1)))
                x2 = max(0, min(img_w-1, int(x2)))
                y2 = max(0, min(img_h-1, int(y2)))
                if x2 > x1 and y2 > y1:
                    loc_pred_formatted.append([x1, y1, x2, y2])
    
    # Calculate stats for all three tasks
    binary_stats = evaluator.generate_statistics(
        args.dataset, 'binary', gt, binary_pred, image_size=image.size
    )
    
    loc_stats = evaluator.generate_statistics(
        args.dataset, 'localization', gt, loc_pred_formatted, image_size=image.size
    )
    
    legion_stats = legion_evaluator.generate_statistics(
        args.dataset, 'localization', gt, loc_pred_formatted, image_size=image.size
    )
    wsol_stats = wsol_evaluator.generate_statistics(
        args.dataset, 'localization', gt, loc_pred_formatted, image_size=image.size
    )
    
    expl_stats = evaluator.generate_statistics(
        args.dataset, 'explanation', gt, expl_pred, image_size=image.size
    )
    
    # Collect metrics
    binary_success = binary_stats.get('binary_success', False)
    pred_positive = binary_pred.get('prediction', False)
    gt_positive = has_gt
    
    if pred_positive and gt_positive:
        classification = 'TP'
    elif pred_positive and not gt_positive:
        classification = 'FP'
    elif not pred_positive and not gt_positive:
        classification = 'TN'
    elif not pred_positive and gt_positive:
        classification = 'FN'
    
    # Collect localization and explanation metrics (only for positive samples)
    if has_gt:
        iou = legion_stats.get('iou', 0.0) if legion_stats.get('iou') is not None else 0.0
        pixel_f1 = legion_stats.get('pixel_f1', 0.0) if legion_stats.get('pixel_f1') is not None else 0.0
        pixel_precision = legion_stats.get('pixel_precision', 0.0) if legion_stats.get('pixel_precision') is not None else 0.0
        pixel_recall = legion_stats.get('pixel_recall', 0.0) if legion_stats.get('pixel_recall') is not None else 0.0
        rouge_l = expl_stats.get('rouge_l', 0.0) if expl_stats.get('rouge_l') is not None else 0.0
        css = expl_stats.get('css', 0.0) if expl_stats.get('css') is not None else 0.0
    else:
        iou = pixel_f1 = pixel_precision = pixel_recall = 0.0
        rouge_l = css = None
    
    # Log prediction vs GT
    logger.info(f"\n{'='*80}")
    if args.dataset in ['val', 'train']:
        uuid = gt.get('uuid', 'unknown')
        image_type = gt.get('image_type', 'unknown')
        num_artifacts = len(gt.get('artifacts', []))
        logger.info(f"Sample {total_processed}: {image_path.name} (UUID: {uuid}, Type: {image_type}, GT Artifacts: {num_artifacts})")
    else:
        logger.info(f"Sample {total_processed}: {image_path.name}")
    logger.info(f"{'='*80}")
    logger.info(f"BINARY DETECTION:")
    logger.info(f"  GT: {gt_positive} | Pred: {pred_positive} | Classification: {classification}")
    logger.info(f"\nLOCALIZATION:")
    logger.info(f"  GT bboxes: {len(gt.get('bboxes', []))} artifacts")
    logger.info(f"  Pred bboxes: {len(loc_pred)} detections")
    if has_gt:
        logger.info(f"  LEGION IoU: {iou:.4f}, LEGION F1: {pixel_f1:.4f}")
    
    logger.info(f"\nEXPLANATION:")
    gt_explanation = gt.get('explanation', 'N/A')
    pred_explanation = expl_pred.get('explanation', 'N/A')
    logger.info(f"  GT: {gt_explanation[:150]}{'...' if len(gt_explanation) > 150 else ''}")
    logger.info(f"  Pred: {pred_explanation[:150]}{'...' if len(pred_explanation) > 150 else ''}")
    if has_gt:
        logger.info(f"  ROUGE-L: {rouge_l:.4f}, CSS: {css:.4f}")
    logger.info(f"{'='*80}\n")
    
    # Store comprehensive result
    result_entry = {
        'image_path': str(image_path),
        'ground_truth': gt,
        'predictions': {
            'binary': binary_pred,
            'localization': loc_pred,
            'explanation': expl_pred
        },
        'raw_outputs': {
            'binary': binary_output,
            'localization': loc_output,
            'explanation': expl_output
        },
        'binary_success': binary_success,
        'classification': classification,
        'iou': loc_stats.get('iou') if has_gt else None,
        'loc_f1': loc_stats.get('loc_f1') if has_gt else None,
        'loc_precision': loc_stats.get('loc_precision') if has_gt else None,
        'loc_recall': loc_stats.get('loc_recall') if has_gt else None,
        'legion_iou': iou if has_gt else None,
        'legion_pixel_f1': pixel_f1 if has_gt else None,
        'legion_pixel_precision': pixel_precision if has_gt else None,
        'legion_pixel_recall': pixel_recall if has_gt else None,
        'wsol_iou': wsol_stats.get('iou') if has_gt else None,
        'rouge_l': rouge_l if has_gt else None,
        'css': css if has_gt else None,
        'has_gt_artifacts': has_gt
    }
    
    return result_entry, binary_success, classification, iou, pixel_f1, pixel_precision, pixel_recall, rouge_l, css, has_gt


def run_multi_task_vqa_evaluation(args):
    """Run evaluation for multi-task VQA format with batch support."""
    
    # Set dataset-specific path
    if args.dataset in ['val', 'train']:
        dataset_path = args.dataset_path
    elif args.dataset == 'synthscars':
        dataset_path = "/data2/jhpark/image-artifacts/data/eval/SynthScars/test"
    elif args.dataset == 'synartifact':
        dataset_path = "/data2/jhpark/image-artifacts/data/eval/SynArtifact"
    elif args.dataset == 'loki':
        dataset_path = "/data2/jhpark/image-artifacts/data/eval/loki"
    elif args.dataset == 'richhf':
        dataset_path = "/data2/jhpark/image-artifacts/data/eval/richhf-18k"
    elif args.dataset == 'ours':
        dataset_path = "/data2/jhpark/image-artifacts/data/eval/ours"
    else:
        dataset_path = "/data2/jhpark/image-artifacts/data/eval"
    
    # Setup logging
    exp_name = Path(args.exp_dir).name if args.exp_dir else args.model_type
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "eval_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), args.dataset, args.model_type, exp_name)
    
    logger.info("=" * 80)
    logger.info("MULTI-TASK VQA EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model type: {args.model_type}")
    if args.exp_dir:
        logger.info(f"Experiment directory: {args.exp_dir}")
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info(f"Dataset path: {dataset_path}")
    if args.dataset in ['val', 'train']:
        logger.info(f"Dataset JSON file: {args.dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    logger.info("")
    logger.info("Evaluating three independent tasks:")
    logger.info("  1. Binary Detection")
    logger.info("  2. Localization")
    logger.info("  3. Global Explanation")
    if args.batch_size > 1:
        logger.info(f"Note: Using batch inference with size {args.batch_size}")
    logger.info("=" * 80)
    
    # Setup configuration
    config = {
        'model_type': args.model_type,
        'model_path': args.exp_dir,
        'device': args.device,
        'dataset_type': args.dataset,
        'base_dir': dataset_path,
        'dataset_path': dataset_path if args.dataset in ['val', 'train'] else None,
        'use_finetuned': bool(args.exp_dir),  # True if exp_dir is provided
        'finetune_mode': 'custom',  # Placeholder for models that need it
    }
    
    # Initialize model
    logger.info("Initializing model...")
    model = create_model(config)
    
    # Setup dataset
    logger.info("Loading dataset...")
    data_iterator = DatasetIterator(config)
    
    # Setup evaluation metrics
    evaluator = Evaluation()
    legion_evaluator = legion_eval_utils.Evaluation()
    wsol_evaluator = wsol_eval_utils.Evaluation()
    
    # Get three prompts
    prompts = get_prompts()
    logger.info("\nUsing training template prompts:")
    logger.info(f"Binary prompt: {prompts['binary'][:80]}...")
    logger.info(f"Localization prompt: {prompts['localization'][:80]}...")
    logger.info(f"Explanation prompt: {prompts['explanation'][:80]}...")
    logger.info("")
    
    # Initialize metrics for all three tasks
    all_results = []
    all_binary_success = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    all_iou_scores = []
    all_pixel_f1_scores = []
    all_pixel_precision_scores = []
    all_pixel_recall_scores = []
    all_rouge_l_scores = []
    all_css_scores = []
    
    total_processed = 0
    total_samples = args.max_samples if args.max_samples else len(data_iterator)
    pbar = tqdm(total=total_samples, desc="Evaluating samples", unit="sample")
    
    # Batch processing loop
    current_batch_size = args.batch_size
    batch_gts = []
    batch_image_paths = []
    batch_images = []
    
    for gt, image_path in data_iterator:
        if args.max_samples and total_processed >= args.max_samples:
            break
        
        # Load image
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        image = Image.open(image_path).convert('RGB')
        if args.dataset == 'richhf':
            image = image.resize((512, 512), Image.LANCZOS)
        
        batch_gts.append(gt)
        batch_image_paths.append(image_path)
        batch_images.append(image)
        
        # Process batch when it's full or we've reached the end
        if len(batch_images) >= current_batch_size or total_processed + len(batch_images) >= total_samples:
            try:
                if args.batch_size > 1:
                    # Use batch inference
                    binary_outputs = unified_batch_inference(model, batch_images, prompts['binary'])
                    loc_outputs = unified_batch_inference(model, batch_images, prompts['localization'])
                    expl_outputs = unified_batch_inference(model, batch_images, prompts['explanation'])
                else:
                    # Single sample inference
                    binary_outputs = [unified_inference(model, img, prompts['binary']) for img in batch_images]
                    loc_outputs = [unified_inference(model, img, prompts['localization']) for img in batch_images]
                    expl_outputs = [unified_inference(model, img, prompts['explanation']) for img in batch_images]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and len(batch_images) > 1:
                    logger.warning("OOM during batched inference. Falling back to per-sample inference.")
                    current_batch_size = max(1, current_batch_size // 2)
                    binary_outputs = []
                    loc_outputs = []
                    expl_outputs = []
                    for img in batch_images:
                        try:
                            binary_outputs.append(unified_inference(model, img, prompts['binary']))
                            loc_outputs.append(unified_inference(model, img, prompts['localization']))
                            expl_outputs.append(unified_inference(model, img, prompts['explanation']))
                        except Exception as inner_e:
                            logger.error(f"Per-sample inference failed: {inner_e}")
                            binary_outputs.append("")
                            loc_outputs.append("")
                            expl_outputs.append("")
                else:
                    raise
            
            # Process each sample in the batch
            for gt, image_path, image, binary_output, loc_output, expl_output in zip(
                batch_gts, batch_image_paths, batch_images, binary_outputs, loc_outputs, expl_outputs
            ):
                total_processed += 1
                
                result_entry, binary_success, classification, iou, pixel_f1, pixel_precision, pixel_recall, rouge_l, css, has_gt = process_sample(
                    args, gt, image_path, image, binary_output, loc_output, expl_output,
                    evaluator, legion_evaluator, wsol_evaluator, total_processed
                )
                
                # Update metrics
                all_binary_success.append(binary_success)
                all_results.append(result_entry)
                
                if classification == 'TP':
                    true_positives += 1
                elif classification == 'FP':
                    false_positives += 1
                elif classification == 'TN':
                    true_negatives += 1
                elif classification == 'FN':
                    false_negatives += 1
                
                if has_gt:
                    all_iou_scores.append(iou)
                    all_pixel_f1_scores.append(pixel_f1)
                    all_pixel_precision_scores.append(pixel_precision)
                    all_pixel_recall_scores.append(pixel_recall)
                    if rouge_l is not None:
                        all_rouge_l_scores.append(rouge_l)
                    if css is not None:
                        all_css_scores.append(css)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Binary': f'{binary_success}',
                    'LEGION_IoU': f'{iou:.3f}' if has_gt else '0.000',
                    'ROUGE': f'{rouge_l:.3f}' if rouge_l is not None else '0.000'
                })
            
            # Clear batch for next iteration
            batch_gts = []
            batch_image_paths = []
            batch_images = []
    
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
    
    f1_metrics = calculate_f1_metrics(true_positives, false_positives, true_negatives, false_negatives)
    logger.info(f"  Precision: {f1_metrics['precision']:.4f} ({f1_metrics['precision']*100:.2f}%)")
    logger.info(f"  Recall: {f1_metrics['recall']:.4f} ({f1_metrics['recall']*100:.2f}%)")
    logger.info(f"  F1 Score: {f1_metrics['f1']:.4f} ({f1_metrics['f1']*100:.2f}%)")
    logger.info(f"  Confusion Matrix: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")
    logger.info(f"  Total samples: {len(all_binary_success)}")
    
    # Localization
    logger.info("\nLOCALIZATION (LEGION Metrics):")
    if all_iou_scores:
        mean_iou = sum(all_iou_scores) / len(all_iou_scores)
        mean_f1 = sum(all_pixel_f1_scores) / len(all_pixel_f1_scores)
        mean_precision = sum(all_pixel_precision_scores) / len(all_pixel_precision_scores)
        mean_recall = sum(all_pixel_recall_scores) / len(all_pixel_recall_scores)
        
        logger.info(f"  Mean LEGION IoU: {mean_iou:.4f}")
        logger.info(f"  Mean LEGION F1: {mean_f1:.4f}")
        logger.info(f"  Mean LEGION Precision: {mean_precision:.4f}")
        logger.info(f"  Mean LEGION Recall: {mean_recall:.4f}")
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
        logger.info(f"  Valid samples (GT positive): {len(all_rouge_l_scores)}")
    else:
        logger.info("  No explanation samples found")
        mean_rouge = mean_css = 0.0
    
    logger.info("=" * 80)
    
    # Save results
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"multi_task_vqa_{args.dataset}_{args.model_type}_{exp_name}_{timestamp}.json"
    
    metrics = {
        'binary': {
            'accuracy': binary_acc,
            'precision': f1_metrics['precision'],
            'recall': f1_metrics['recall'],
            'f1': f1_metrics['f1'],
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            },
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
            'valid_samples': len(all_rouge_l_scores)
        }
    }
    
    final_results = {
        'config': {
            'model_type': args.model_type,
            'exp_dir': args.exp_dir,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'output_dir': str(results_dir),
            'format': 'multi_task_vqa',
            'timestamp': timestamp
        },
        'metrics': metrics,
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    logger.info(f"✅ Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Task VQA Evaluation with Batch Support")
    parser.add_argument("--model-type", type=str, default="qwen", 
                        choices=["qwen", "intern", "gpt", "gemini", "pal", "diff", "legion"],
                        help="Model type to use for evaluation")
    parser.add_argument("--exp-dir", type=str, default=None, help="Path to experiment directory (model checkpoint)")
    parser.add_argument("--dataset", type=str, default="ours", 
                        choices=["ours", "synthscars", "synartifact", "loki", "richhf", "val", "train"], 
                        help="Dataset to evaluate")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Path to dataset JSON file (required when --dataset val or train)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save evaluation results (default: ./eval_logs)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset in ["val", "train"] and not args.dataset_path:
        parser.error(f"--dataset-path is required when --dataset {args.dataset}")
    
    run_multi_task_vqa_evaluation(args)


if __name__ == "__main__":
    main()
