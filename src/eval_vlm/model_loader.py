"""
Model loading utilities for finetuned VLM models.

This module provides functions to load finetuned models from experiment directories
and handle different model architectures.
"""

import os
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    AutoTokenizer,
    AutoConfig
)
from qwen_vl_utils import process_vision_info


class FinetunedModelLoader:
    """
    Loader for finetuned vision-language models from experiment directories.
    """
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        """
        Initialize the model loader.
        
        Args:
            exp_dir: Path to the experiment directory containing the finetuned model
            device: Device to load the model on
        """
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Validate experiment directory
        self._validate_exp_dir()
        
    def _validate_exp_dir(self):
        """Validate that the experiment directory contains a valid model."""
        if not self.exp_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {self.exp_dir}")
        
        # Check for common model files
        required_files = ["config.json"]
        model_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
        
        config_file = self.exp_dir / "config.json"
        if not config_file.exists():
            raise ValueError(f"config.json not found in {self.exp_dir}")
        
        # Check for at least one model file
        has_model_file = any((self.exp_dir / f).exists() for f in model_files)
        if not has_model_file:
            raise ValueError(f"No model files found in {self.exp_dir}")
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the finetuned model and processor.
        
        Returns:
            Dictionary containing model, processor, and tokenizer
        """
        try:
            # Load configuration to determine model type
            config = AutoConfig.from_pretrained(str(self.exp_dir), trust_remote_code=True)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.exp_dir), 
                trust_remote_code=True
            )
            self.tokenizer.padding_side = "left"
            
            # Load model based on architecture
            if "qwen" in config.architectures[0].lower():
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(self.exp_dir),
                    trust_remote_code=True,
                    device_map=self.device,
                    torch_dtype=torch.float16
                )
            else:
                # Fallback for other architectures
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(self.exp_dir),
                    trust_remote_code=True,
                    device_map=self.device,
                    torch_dtype=torch.float16
                )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(str(self.exp_dir))
            self.processor.tokenizer.padding_side = "left"
            
            return {
                "model": self.model,
                "processor": self.processor,
                "tokenizer": self.tokenizer,
                "config": config
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.exp_dir}: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            "exp_dir": str(self.exp_dir),
            "device": self.device,
            "model_type": type(self.model).__name__,
            "config": self.model.config if hasattr(self.model, 'config') else None
        }


def load_finetuned_model(exp_dir: str, device: str = "cuda:0") -> Dict[str, Any]:
    """
    Convenience function to load a finetuned model.
    
    Args:
        exp_dir: Path to the experiment directory
        device: Device to load the model on
        
    Returns:
        Dictionary containing model, processor, and tokenizer
    """
    loader = FinetunedModelLoader(exp_dir, device)
    return loader.load_model()


def detect_model_type(exp_dir: str) -> str:
    """
    Detect the model type from the experiment directory.
    
    Args:
        exp_dir: Path to the experiment directory
        
    Returns:
        Model type string (e.g., 'qwen2_5vl', 'llava', etc.)
    """
    config_path = Path(exp_dir) / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {exp_dir}")
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        architectures = config.get('architectures', [])
        if architectures:
            arch = architectures[0].lower()
            if 'qwen' in arch:
                return 'qwen2_5vl'
            elif 'llava' in arch:
                return 'llava'
            elif 'llama' in arch:
                return 'llama'
            else:
                return 'unknown'
        else:
            return 'unknown'
            
    except Exception as e:
        print(f"Warning: Could not detect model type: {e}")
        return 'unknown'
