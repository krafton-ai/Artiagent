"""
Image Artifacts Pipeline

This module provides detection and processing pipeline components for generating
image artifacts using various segmentation models.
"""

# Core detection models
try:
    # from .vlpart_detector import VLPartDetector
    from .gsam_detector import GSAMDetector 
except:
    pass

try:
    from .flux_generator import FluxGenerator, FluxConfig
except:
    pass


from .data_loader import COCODataLoader, ImageNetDataLoader, CustomDirectoryDataLoader
from .instance_processor import InstanceProcessor
from .visualization import ImageVisualizer


__all__ = [
    'GSAMDetector',
    'FluxGenerator',
    'FluxConfig',
    'COCODataLoader',
    'ImageNetDataLoader',
    'CustomDirectoryDataLoader',
    'InstanceProcessor',
    'ImageVisualizer',
]