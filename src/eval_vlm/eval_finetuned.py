"""
Main evaluation script for finetuned VLM models.

This script evaluates finetuned vision-language models on various artifact detection
datasets. It accepts any experiment directory containing a finetuned model checkpoint.
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from pathlib import Path

# Add the parent eval directory to the path to import existing utilities
sys.path.append(str(Path(__file__).parent.parent / "eval"))

from model_loader import FinetunedModelLoader, detect_model_type
from eval_utils import Evaluation, parse_json, create_prompt
import legion_eval_utils
import wsol_eval_utils
from qwen_vl_utils import process_vision_info


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
            return {'prediction': True}
            
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
        # Look for the pattern: [x1, y1, x2, y2]: description
        bbox_pattern = re.compile(r'\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*:')
        match = bbox_pattern.search(raw_output)
        
        if match:
            explanation_text = raw_output[:match.start()].strip()
            # Remove any trailing incomplete bracket or punctuation
            explanation_text = re.sub(r'[\[\(]\s*$', '', explanation_text).strip()
        else:
            explanation_text = raw_output.strip()
            # Remove any trailing incomplete bracket or punctuation
            explanation_text = re.sub(r'[\[\(]\s*$', '', explanation_text).strip()
            
        return {"explanation": explanation_text}
    
    # Fallback to raw output
    return {"raw_response": raw_output}


def setup_logging(output_dir: str, dataset_type: str, exp_dir: str, eval_type: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        dataset_type: Dataset name for log file naming
        exp_dir: Experiment directory path for log file naming
        eval_type: Evaluation type for log file naming
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = Path(exp_dir).name
    
    if eval_type == 'localization':
        log_file = log_dir / f'finetuned_eval_{dataset_type}_{exp_name}_bbox_{timestamp}.log'
    else:
        log_file = log_dir / f'finetuned_eval_{dataset_type}_{exp_name}_{eval_type}_{timestamp}.log'
    
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
            image_path = self.base_dir / "ours" / f"images/{json_data['id']}.png"
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
        json_path = self.base_dir / "ours" / "metadata.json"
        with open(json_path, "r") as f:
            self.data = json.load(f)


class FinetunedModelEvaluator:
    """
    Evaluator for finetuned VLM models.
    """
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        """
        Initialize the finetuned model evaluator.
        
        Args:
            exp_dir: Path to the experiment directory containing the finetuned model
            device: Device to run inference on
        """
        self.exp_dir = exp_dir
        self.device = device
        self.model_loader = FinetunedModelLoader(exp_dir, device)
        self.model_components = None
        
    def load_model(self):
        """Load the finetuned model and components."""
        self.model_components = self.model_loader.load_model()
        self.model = self.model_components["model"]
        self.processor = self.model_components["processor"]
        self.tokenizer = self.model_components["tokenizer"]
    
    def unified_prompt(self):
        """
        Create a unified prompt that asks for all three tasks simultaneously.
        This function provides a template that can be manually modified.
        """
        prompt = "Analyze the image and describe any visual anomalies. Provide bounding boxes and explain in detail."
        return prompt
        
    def inference(self, image: Image.Image, prompt: str) -> str:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for inference
            
        Returns:
            Raw text output from the model
        """
        if self.model_components is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare the conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # INSERT_YOUR_CODE
        print(f"Image size: {image.size}")
        # Process the input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(text)
        
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of PIL Images to analyze
            prompt: Text prompt for inference
            
        Returns:
            List of raw text outputs from the model
        """
        if self.model_components is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare the conversation for each image
        messages_list = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            messages_list.append(messages)
        
        # Process all inputs
        texts = []
        image_inputs_list = []
        for messages in messages_list:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)
        
        # Process all inputs together
        inputs = self.processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Generate responses for all images
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_texts


def run_batch_evaluation(config: Dict, max_samples: Optional[int] = None):
    """
    Run batch evaluation on dataset with finetuned model.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    eval_type = config['eval_type']
    exp_dir = config['exp_dir']
    batch_size = config.get('batch_size', 2)
    
    logger.info(f"Starting batch evaluation for {dataset_type} dataset")
    logger.info(f"Using finetuned model from: {exp_dir}")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    evaluator_model = FinetunedModelEvaluator(exp_dir, config['device'])
    evaluator_model.load_model()
    
    data_iterator = DatasetIterator(config)
    evaluator = Evaluation()
    
    # Create additional evaluators for comprehensive localization evaluation
    legion_evaluator = legion_eval_utils.Evaluation()
    wsol_evaluator = wsol_eval_utils.Evaluation()
    
    # Determine number of samples to process
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    logger.info(f"Processing {total_samples} samples in batches of {batch_size}")

    results = {}
    processed = 0

    # Create prompt based on configuration
    if config.get('prompt_match', False):
        prompt = evaluator_model.unified_prompt()
        logger.info(f"Using unified prompt: {prompt}")
    else:
        prompt = create_prompt(eval_type)
        logger.info(f"Input query: {prompt}")

    try:
        while True:
            if max_samples and processed >= max_samples:
                break
                
            # Collect a batch of samples
            batch_json_data = []
            batch_image_paths = []
            batch_images = []
            
            while len(batch_images) < batch_size:
                try:
                    json_data, image_path = next(data_iterator)
                except StopIteration:
                    break
                    
                if not image_path.exists():
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
                
            logger.info(f"Processing batch starting at index {processed + 1} with size {len(batch_images)}")
            
            # Run batched inference with OOM fallback
            try:
                batch_raw_outputs = evaluator_model.inference_batch(batch_images, prompt)
                batch_predictions = [process_finetuned_output(output, eval_type) for output in batch_raw_outputs]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and len(batch_images) > 1:
                    logger.warning("OOM during batched inference. Falling back to per-sample inference for this batch.")
                    batch_predictions = []
                    batch_raw_outputs = []
                    for img in batch_images:
                        try:
                            raw_output = evaluator_model.inference(img, prompt)
                            batch_raw_outputs.append(raw_output)
                            batch_predictions.append(process_finetuned_output(raw_output, eval_type))
                        except Exception as inner_e:
                            logger.error(f"Per-sample inference failed: {inner_e}")
                            batch_predictions.append({"error": str(inner_e)})
                            batch_raw_outputs.append("")
                else:
                    raise
            
            # Evaluate each item in the batch
            for idx, (json_data, image_path, image, prediction, raw_output) in enumerate(
                zip(batch_json_data, batch_image_paths, batch_images, batch_predictions, batch_raw_outputs)
            ):
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
                
                # Create sample result based on evaluation type
                if eval_type == 'binary':
                    sample_result = {
                        'image_path': str(image_path),
                        'binary_success': stats['binary_success'],
                        'classification': stats['classification'],
                        'has_gt_artifacts': stats['has_gt_artifacts'],
                        'has_pred_artifacts': stats['has_pred_artifacts'],
                        'prediction': prediction,
                        'raw_output': raw_output
                    }
                    logger.info(
                        f"Sample {processed + idx + 1} - Binary: {sample_result['binary_success']}, "
                        f"Prediction: {prediction}"
                    )
                elif eval_type == 'localization':
                    if dataset_type == 'synartifact':
                        has_gt_artifacts = bool(json_data.get('Artifacts annotation', []))
                    elif dataset_type == 'ours':
                        has_gt_artifacts = json_data.get('has_artifacts', False)
                    else:
                        has_gt_artifacts = True
                        
                    if has_gt_artifacts:
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
                            'prediction': prediction,
                            'raw_output': raw_output
                        }
                        logger.info(
                            f"Sample {processed + idx + 1} - IoU: {sample_result['iou']:.3f}, "
                            f"F1: {sample_result['loc_f1']:.3f} (P: {sample_result['loc_precision']:.3f}, "
                            f"R: {sample_result['loc_recall']:.3f}, TP/FP/FN: {sample_result['loc_tp']}/{sample_result['loc_fp']}/{sample_result['loc_fn']})"
                        )
                    else:
                        sample_result = {
                            'image_path': str(image_path),
                            # Standard evaluation metrics
                            'iou': None,
                            'loc_tp': None,
                            'loc_fp': None,
                            'loc_fn': None,
                            'loc_precision': None,
                            'loc_recall': None,
                            'loc_f1': None,
                            # LEGION evaluation metrics
                            'legion_iou': None,
                            'legion_miou': None,
                            'legion_iou_foreground': None,
                            'legion_iou_background': None,
                            'legion_pixel_f1': None,
                            'legion_pixel_precision': None,
                            'legion_pixel_recall': None,
                            # WSOL evaluation metrics
                            'wsol_iou': None,
                            'prediction': prediction,
                            'raw_output': raw_output
                        }
                        logger.info(f"Sample {processed + idx + 1} - Skipped (negative sample)")
                        
                elif eval_type == 'explanation':
                    sample_result = {
                        'image_path': str(image_path),
                        'rouge_l': stats['rouge_l'],
                        'css': stats['css'],
                        'prediction': prediction,
                        'raw_output': raw_output
                    }
                    logger.info(
                        f"Sample {processed + idx + 1} - ROUGE-L: {sample_result['rouge_l']:.3f}, "
                        f"CSS: {sample_result['css']:.3f}"
                    )
                else:
                    raise ValueError(f"Unsupported evaluation type: {eval_type}")

                results[processed + idx] = sample_result

            processed += len(batch_images)
            
    except StopIteration:
        logger.info("Reached end of dataset")
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
    
    logger.info("Batch evaluation completed!")
    
    # Compute and log summary statistics
    if results:
        _log_summary_statistics(results, eval_type, logger)
    
    return results


def run_evaluation(config: Dict, max_samples: Optional[int] = None):
    """
    Run evaluation on dataset with finetuned model (single sample processing).
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    eval_type = config['eval_type']
    exp_dir = config['exp_dir']
    
    logger.info(f"Starting evaluation for {dataset_type} dataset")
    logger.info(f"Using finetuned model from: {exp_dir}")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    evaluator_model = FinetunedModelEvaluator(exp_dir, config['device'])
    evaluator_model.load_model()
    
    data_iterator = DatasetIterator(config)
    evaluator = Evaluation()
    
    # Create additional evaluators for comprehensive localization evaluation
    legion_evaluator = legion_eval_utils.Evaluation()
    wsol_evaluator = wsol_eval_utils.Evaluation()
    
    # Determine number of samples to process
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    logger.info(f"Processing {total_samples} samples")

    results = {}

    # Create prompt based on configuration
    if config.get('prompt_match', False):
        prompt = evaluator_model.unified_prompt()
        logger.info(f"Using unified prompt: {prompt}")
    else:
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

            # Run model inference
            raw_output = evaluator_model.inference(image, prompt)
            logger.info(f"Model output: {raw_output}")
            
            # Process the output for evaluation
            prediction = process_finetuned_output(raw_output, eval_type)
            logger.info(f"Processed prediction: {prediction}")
            
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
            
            # Create sample result based on evaluation type
            if eval_type == 'binary':
                sample_result = {
                    'image_path': str(image_path),
                    'binary_success': stats['binary_success'],
                    'classification': stats['classification'],
                    'has_gt_artifacts': stats['has_gt_artifacts'],
                    'has_pred_artifacts': stats['has_pred_artifacts'],
                    'prediction': prediction,
                    'raw_output': raw_output
                }
                logger.info(
                    f"Sample {i + 1} - Binary: {sample_result['binary_success']}, "
                    f"Prediction: {prediction}"
                )
            elif eval_type == 'localization':
                if dataset_type == 'synartifact':
                    has_gt_artifacts = bool(json_data.get('Artifacts annotation', []))
                elif dataset_type == 'ours':
                    has_gt_artifacts = json_data.get('has_artifacts', False)
                else:
                    has_gt_artifacts = True
                    
                if has_gt_artifacts:
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
                        'prediction': prediction,
                        'raw_output': raw_output
                    }
                    logger.info(
                        f"Sample {i + 1} - IoU: {sample_result['iou']:.3f}, "
                        f"F1: {sample_result['loc_f1']:.3f} (P: {sample_result['loc_precision']:.3f}, "
                        f"R: {sample_result['loc_recall']:.3f}, TP/FP/FN: {sample_result['loc_tp']}/{sample_result['loc_fp']}/{sample_result['loc_fn']})"
                    )
                else:
                    sample_result = {
                        'image_path': str(image_path),
                        # Standard evaluation metrics
                        'iou': None,
                        'loc_tp': None,
                        'loc_fp': None,
                        'loc_fn': None,
                        'loc_precision': None,
                        'loc_recall': None,
                        'loc_f1': None,
                        # LEGION evaluation metrics
                        'legion_iou': None,
                        'legion_miou': None,
                        'legion_iou_foreground': None,
                        'legion_iou_background': None,
                        'legion_pixel_f1': None,
                        'legion_pixel_precision': None,
                        'legion_pixel_recall': None,
                        # WSOL evaluation metrics
                        'wsol_iou': None,
                        'prediction': prediction,
                        'raw_output': raw_output
                    }
                    logger.info(f"Sample {i + 1} - Skipped (negative sample)")
                    
            elif eval_type == 'explanation':
                sample_result = {
                    'image_path': str(image_path),
                    'rouge_l': stats['rouge_l'],
                    'css': stats['css'],
                    'prediction': prediction,
                    'raw_output': raw_output
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
    
    # Compute and log summary statistics
    if results:
        _log_summary_statistics(results, eval_type, logger)
    
    return results


def _log_summary_statistics(results: Dict, eval_type: str, logger: logging.Logger):
    """Log summary statistics for the evaluation results."""
    total_samples = len(results)
    
    if eval_type == 'binary':
        binary_accuracy = sum(r.get('binary_success', False) for _, r in results.items()) / total_samples
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Binary classification accuracy: {binary_accuracy:.3f}")
        
    elif eval_type == 'localization':
        # Filter out None values for negative samples
        valid_loc_results = [r for _, r in results.items() if r.get('iou') is not None]
        valid_samples = len(valid_loc_results)
        
        if valid_samples > 0:
            mean_iou = sum(r.get('iou', 0.0) for r in valid_loc_results) / valid_samples
            mean_loc_f1 = sum(r.get('loc_f1', 0.0) for r in valid_loc_results) / valid_samples
            mean_loc_precision = sum(r.get('loc_precision', 0.0) for r in valid_loc_results) / valid_samples
            mean_loc_recall = sum(r.get('loc_recall', 0.0) for r in valid_loc_results) / valid_samples
            
            logger.info("=" * 80)
            logger.info("LOCALIZATION EVALUATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Valid samples (positive): {valid_samples}")
            logger.info(f"Mean IoU: {mean_iou:.3f}")
            logger.info(f"Mean F1: {mean_loc_f1:.3f}")
            logger.info(f"Mean Precision: {mean_loc_precision:.3f}")
            logger.info(f"Mean Recall: {mean_loc_recall:.3f}")
        else:
            logger.info("No valid localization samples found")
            
    elif eval_type == 'explanation':
        mean_rouge_l = sum(r.get('rouge_l', 0.0) for _, r in results.items()) / total_samples
        mean_css = sum(r.get('css', 0.0) for _, r in results.items()) / total_samples
        
        logger.info("=" * 60)
        logger.info("EXPLANATION EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Mean ROUGE-L: {mean_rouge_l:.3f}")
        logger.info(f"Mean CSS: {mean_css:.3f}")


def main():
    """Main function for finetuned model evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate finetuned VLM models on artifact detection tasks'
    )
    parser.add_argument('--exp-dir', type=str, required=True,
                       help='Path to the experiment directory containing the finetuned model')
    parser.add_argument('--dataset', type=str, 
                       choices=['synthscars', 'synartifact', 'loki', 'richhf', 'ours'], 
                       default='ours', help='Dataset to evaluate on (default: ours)')
    parser.add_argument('--type', type=str,
                       choices=['binary', 'localization', 'explanation'],
                       default='explanation', help='Evaluation type (default: explanation)')
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
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to dataset (alias for --base-dir)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for evaluation (default: 2)')
    parser.add_argument('--use-batch', action='store_true',
                       help='Use batch processing for evaluation')
    parser.add_argument('--prompt-match', action='store_true',
                       help='Use unified prompt for evaluation instead of type-specific prompt')
    
    args = parser.parse_args()
    
    # Validate experiment directory
    if not Path(args.exp_dir).exists():
        raise ValueError(f"Experiment directory does not exist: {args.exp_dir}")
    
    # Set dataset paths if not provided
    # Use --dataset-path if provided, otherwise use --base-dir
    if args.dataset_path is not None:
        base_dir = args.dataset_path
    elif args.base_dir is not None:
        base_dir = args.base_dir
    else:
        dataset_paths = {
            'synthscars': "/data2/jhpark/image-artifacts/data/eval/SynthScars",
            'synartifact': "/data2/jhpark/image-artifacts/data/eval/SynArtifact",
            'loki': "/data2/jhpark/image-artifacts/data/eval/loki",
            'richhf': "/data2/jhpark/image-artifacts/data/eval/richhf-18k",
            'ours': "/data2/jhpark/image-artifacts/data/eval"
        }
        base_dir = dataset_paths.get(args.dataset)
        if base_dir is None:
            raise ValueError(f"No default path for dataset: {args.dataset}")
    
    # Setup configuration
    config = {
        'exp_dir': args.exp_dir,
        'dataset_type': args.dataset,
        'eval_type': args.type,
        'base_dir': base_dir,
        'log_dir': args.log_dir,
        'device': args.device,
        'batch_size': args.batch_size,
        'prompt_match': args.prompt_match
    }
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.dataset, args.exp_dir, args.type)
    
    # Detect model type
    model_type = detect_model_type(args.exp_dir)
    exp_name = Path(args.exp_dir).name
    
    logger.info(f"🚀 Starting finetuned model evaluation")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"🤖 Model type: {model_type}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"🗒️ Evaluation type: {args.type}")
    logger.info(f"📁 Dataset path: {base_dir}")
    logger.info(f"🔧 Device: {args.device}")

    try:
        # Run evaluation
        if args.use_batch:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            results = run_batch_evaluation(config, args.max_samples)
        else:
            logger.info("Using single sample processing")
            results = run_evaluation(config, args.max_samples)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.type == 'localization':
            results_file = output_dir / f"finetuned_results_{args.dataset}_{exp_name}_bbox_{timestamp}.json"
        else:
            results_file = output_dir / f"finetuned_results_{args.dataset}_{exp_name}_{args.type}_{timestamp}.json"

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
