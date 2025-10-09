"""
Mock LegionEval class that uses pre-generated responses.

This allows running evaluations in the lfac environment using
pre-generated LEGION responses from the legion1.4.7 environment.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Any
from PIL import Image
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MockLegionEval:
    """
    Mock LEGION artifact detector that loads pre-generated responses.
    Compatible with the evaluation scripts' unified_inference interface.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.responses_dir = Path('/data2/jhpark/image-artifacts/eval/legion_responses')
        self.loaded_responses = {}
        self._load_all_responses()

    def _load_all_responses(self):
        """Load all pre-generated responses from pickle files"""
        logger.info("Loading pre-generated LEGION responses...")
        
        datasets = ['synthscars', 'synartifact', 'loki', 'richhf']
        
        for dataset in datasets:
            response_file = self.responses_dir / f"{dataset}_responses.pkl"
            if response_file.exists():
                try:
                    with open(response_file, 'rb') as f:
                        dataset_responses = pickle.load(f)
                    self.loaded_responses[dataset] = dataset_responses
                    logger.info(f"✅ Loaded {len(dataset_responses)} responses for {dataset}")
                except Exception as e:
                    logger.warning(f"Failed to load responses for {dataset}: {e}")
                    self.loaded_responses[dataset] = {}
            else:
                logger.warning(f"Response file not found: {response_file}")
                self.loaded_responses[dataset] = {}
        
        total_responses = sum(len(responses) for responses in self.loaded_responses.values())
        logger.info(f"✅ Total pre-generated responses loaded: {total_responses}")

    def _get_dataset_from_path(self, image_path: str) -> str:
        """Determine dataset from image path"""
        path_str = str(image_path).lower()
        
        if 'synthscars' in path_str:
            return 'synthscars'
        elif 'synartifact' in path_str:
            return 'synartifact'
        elif 'loki' in path_str:
            return 'loki'
        elif 'richhf' in path_str:
            return 'richhf'
        else:
            # Try to infer from config if available
            dataset_type = self.config.get('dataset_type', 'unknown')
            if dataset_type in ['synthscars', 'synartifact', 'loki', 'richhf']:
                return dataset_type
            
            logger.warning(f"Could not determine dataset from path: {image_path}")
            return 'unknown'

    def _get_image_key(self, image_path: str) -> str:
        """Extract image filename to use as key"""
        return Path(image_path).name

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        """
        Mock inference that returns pre-generated response for the image.
        
        Args:
            image: PIL Image (used to determine which response to load)
            
        Returns:
            Dictionary with 'heatmap' and 'explanation' keys (compatible with LegionEval)
        """
        # Try to get the image path from the current evaluation context
        # This is a bit of a hack - we'll try to match based on image properties
        return self._get_fallback_response()

    def inference_with_path(self, image: Image.Image, image_path: str) -> Dict[str, Any]:
        """
        Mock inference with explicit image path for accurate response lookup.
        
        Args:
            image: PIL Image 
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'heatmap' and 'explanation' keys
        """
        dataset = self._get_dataset_from_path(image_path)
        image_key = self._get_image_key(image_path)
        
        if dataset in self.loaded_responses and image_key in self.loaded_responses[dataset]:
            response_data = self.loaded_responses[dataset][image_key]['response']
            
            # Convert numpy arrays back to torch tensors if needed
            if response_data.get('heatmap') is not None and isinstance(response_data['heatmap'], np.ndarray):
                response_data = response_data.copy()
                response_data['heatmap'] = torch.from_numpy(response_data['heatmap'])
            
            logger.debug(f"Retrieved pre-generated response for {image_key} from {dataset}")
            return response_data
        else:
            logger.warning(f"No pre-generated response found for {image_key} in {dataset}")
            return self._get_fallback_response()

    def _get_fallback_response(self) -> Dict[str, Any]:
        """Return a fallback response when pre-generated response is not available"""
        return {
            "heatmap": torch.zeros((512, 512), dtype=torch.int), 
            "explanation": "No pre-generated response available for this image.",
            "error": "missing_pregenerated_response"
        }

    def inference_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Batch inference using individual inference calls"""
        return [self.inference(img) for img in images]


# Patch for the existing evaluation scripts
_original_legion_eval = None

def patch_legion_eval():
    """Patch LegionEval in models module to use MockLegionEval"""
    global _original_legion_eval
    
    try:
        import models
        _original_legion_eval = getattr(models, 'LegionEval', None)
        models.LegionEval = MockLegionEval
        logger.info("✅ Patched LegionEval with MockLegionEval")
        return True
    except Exception as e:
        logger.error(f"Failed to patch LegionEval: {e}")
        return False

def unpatch_legion_eval():
    """Restore original LegionEval"""
    global _original_legion_eval
    
    if _original_legion_eval is not None:
        try:
            import models
            models.LegionEval = _original_legion_eval
            logger.info("✅ Restored original LegionEval")
        except Exception as e:
            logger.error(f"Failed to restore LegionEval: {e}")


# Context manager for easy patching
class MockLegionContext:
    """Context manager for using mock LEGION evaluation"""
    
    def __enter__(self):
        patch_legion_eval()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        unpatch_legion_eval()
