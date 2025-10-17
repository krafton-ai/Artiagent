"""
Utility functions for batch evaluation of finetuned VLMs.

This module provides helper functions needed by eval_finetuned_batch.py:
- DatasetIterator: Iterator for different dataset types
- process_finetuned_output: Process model outputs for evaluation
- setup_logging: Configure logging for evaluation runs
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image


def extract_bboxes(text: str) -> List[List[int]]:
    """Extract bounding boxes from text output."""
    pattern = re.compile(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')
    matches = pattern.findall(text)
    
    bboxes = []
    for match in matches:
        x1, y1, x2, y2 = match
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


def setup_logging(output_dir: str, dataset_type: str, exp_dir: str, eval_type: str) -> str:
    """
    Setup logging configuration for evaluation.
    
    Args:
        output_dir: Directory to save log files
        dataset_type: Type of dataset being evaluated
        exp_dir: Experiment directory name
        eval_type: Type of evaluation
        
    Returns:
        Path to the log file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"finetuned_eval_{dataset_type}_{Path(exp_dir).name}_{eval_type}_{timestamp}.log"
    log_file = output_path / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return str(log_file)


class DatasetIterator:
    """
    Iterator for processing different artifact detection datasets.
    
    Supports multiple dataset types with unified interface for batch processing.
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
            
            image_path = self.base_dir / "data" / sample
            json_file = f"data/{root_folder}/annotation_json_artifacts_class/{image_id}.json"
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
            filename = json_data["filename"]
            # RichHF has images in subdirectories (000, 001, 002, etc.)
            # The filename in JSON is like "test/bf07713d-b61a-4323-9515-7e9c4a70253b.png"
            # But actual path is "test/000/bf07713d-b61a-4323-9515-7e9c4a70253b.png"
            image_name = filename.split('/')[-1]  # Get just the filename
            test_dir = self.base_dir / "test"
            
            # Find the subdirectory containing this image
            for subdir in test_dir.iterdir():
                if subdir.is_dir():
                    potential_path = subdir / image_name
                    if potential_path.exists():
                        image_path = potential_path
                        break
            else:
                # Fallback to original path if not found
                image_path = self.base_dir / filename
            
            return json_data, image_path

        elif self.dataset_type == "ours":
            json_data = sample
            image_path = self.base_dir / "ours" / f"images/{json_data['id']}.png"
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
        with open(json_path, "r") as f:
            self.data = json.load(f)
    
    def _load_synartifact(self):
        """Load SynArtifact dataset."""
        eval_path = self.base_dir / "data" / "eval.txt"
        self.data = []
        with open(eval_path, "r") as f:
            for line in f:
                self.data.append(line.strip())
    
    def _load_loki(self):
        """Load LOKI dataset."""
        json_path = self.base_dir / "open_ended_vqa.json"
        with open(json_path, "r", encoding="utf-16") as f:
            self.data = json.load(f)
    
    def _load_richhf(self):
        """Load RichHF dataset."""
        json_path = self.base_dir / "test.json"
        with open(json_path, "r") as f:
            data_dict = json.load(f)
            # Convert dictionary to list of values
            self.data = list(data_dict.values())
    
    def _load_ours(self):
        """Load our custom dataset."""
        json_path = self.base_dir / "ours" / "metadata.json"
        with open(json_path, "r") as f:
            self.data = json.load(f)
            
    def _load_val_set(self):
        """Load the validation set used for training"""
        json_path = self.base_dir
        with open(json_path, "r") as f:
            self.data = json.load(f)
