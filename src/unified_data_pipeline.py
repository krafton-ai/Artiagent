#!/usr/bin/env python3
"""
Unified Data Pipeline for GSAM and FLUX Generated Datasets

This script processes GSAM and FLUX generated data in two sequential phases:
1. LPIPS Filtering Phase: Filters distortion artifacts using LPIPS similarity
2. Explanation Generation Phase: Generates explanations for all passing artifacts

IMPORTANT: Uses ALL-OR-NOTHING approach - if ANY artifact in an experiment fails
filtering or explanation generation, the ENTIRE experiment is discarded.

The pipeline saves both successful experiments and detailed information about
discarded experiments (with failure reasons) to separate files.

Usage:
    python unified_data_pipeline.py --gsam_dir <path> --flux_dir <path> --output_dir <path>
"""

import os
import sys
import glob
import pickle
import argparse
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import query functions from prompts
from pipeline.prompts import artifact_description, artifact_explanation, negative_explanation, MoneyManager
from openai import OpenAI
import lpips



def setup_logging(output_dir: str):
    """Setup thread-safe logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'unified_pipeline_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class UnifiedDataPipeline:
    """Unified pipeline for filtering and generating explanations for GSAM and FLUX datasets"""
    
    def __init__(self, gsam_dir: str, flux_dir: str, output_dir: str):
        """
        Initialize the unified data pipeline.
        
        Args:
            gsam_dir: Directory containing GSAM generated data
            flux_dir: Directory containing FLUX generated data  
            output_dir: Directory to save filtered datasets with explanations
        """
        self.gsam_dir = Path(gsam_dir)
        self.flux_dir = Path(flux_dir)
        self.output_dir = Path(output_dir)
        
        # Validate directories
        if not self.gsam_dir.exists():
            raise ValueError(f"GSAM directory does not exist: {gsam_dir}")
        if not self.flux_dir.exists():
            raise ValueError(f"FLUX directory does not exist: {flux_dir}")
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI()
        self.money_manager = MoneyManager(model="gpt-4o")
        self.experiment_data = {}
        self.filtered_results = {}
        self.lpips_filter_results = {}  # Will be populated in Phase 1
        self.kernel_stats = {}
        self.discarded_experiments = {}  # Track discarded experiments with failure reasons
        
        # Thread safety
        self.results_lock = threading.Lock()
        self.summary_lock = threading.Lock()
        self.discarded_lock = threading.Lock()
        
        # Initialize LPIPS for Phase 1
        self._init_lpips()

    def _init_lpips(self):
        """Initialize LPIPS for distortion filtering"""
        self.lpips_model = lpips.LPIPS(net='alex')
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lpips_model.to(device)
        print("LPIPS initialized successfully")
    
    def find_matching_experiments(self) -> Dict[str, Dict]:
        """Find matching directories between GSAM and FLUX experiments"""
        # Only consider directories that start with "image"
        gsam_image_dirs = {d.name: d for d in self.gsam_dir.iterdir() 
                            if d.is_dir() and not d.name.startswith('log')}
        flux_image_dirs = {d.name: d for d in self.flux_dir.iterdir()
                            if d.is_dir() and not d.name.startswith('log')}
        
        # Find exact matches between image directories
        matching_ids = set(gsam_image_dirs.keys()) & set(flux_image_dirs.keys())
        
        print(f"GSAM image directories: {sorted(gsam_image_dirs.keys())}")
        print(f"FLUX image directories: {sorted(flux_image_dirs.keys())}")
        print(f"Found matching image directories: {sorted(matching_ids)}")
        
        self.experiment_data = {}
        
        # Process matching image directories
        for exp_id in list(matching_ids):
            gsam_dir = gsam_image_dirs[exp_id]
            flux_dir = flux_image_dirs[exp_id]
            
            # Get specific files for GSAM
            metadata_path = gsam_dir / 'metadata.pkl'
            real_image_path = gsam_dir / 'real_image.png'
            
            # Get specific files for FLUX
            artifact_path = flux_dir / 'artifact.png'
            
            # Only include experiments where all required files exist
            if (metadata_path.exists() and real_image_path.exists() and 
                artifact_path.exists()):
                
                self.experiment_data[exp_id] = {
                    'metadata_path': metadata_path,
                    'real_image_path': real_image_path,
                    'artifact_path': artifact_path,
                    'flux_dir': flux_dir
                }
            else:
                missing_files = []
                if not metadata_path.exists():
                    missing_files.append('metadata.pkl')
                if not real_image_path.exists():
                    missing_files.append('real_image.png')
                if not artifact_path.exists():
                    missing_files.append('artifact.png')
                print(f"Skipping {exp_id}: missing files {missing_files}")
        
        total_matches = len(self.experiment_data)
        print(f"Found {total_matches} total matching experiments with all required files")
        print(f"Experiment IDs: {sorted(self.experiment_data.keys())}")
        
        return self.experiment_data
    
    def check_distortion_with_lpips(self, artifact: Dict, artifact_type: str,
                                    orig_img: np.ndarray, img: np.ndarray, exp_id: str, logger) -> Tuple[bool, float]: 
        """
        Use LPIPS to check if distortion artifact is present.
        Returns tuple of (passed, similarity_score) where passed is True if distortion is detected (dissimilar enough).
        """
        # Set default thresholds
        thresholds = {
            'similar': 0.9,  # If similarity > 0.9, images are too similar (no distortion)
            'different': 0.5,  # If similarity < 0.5, images are too different (distortion)
        }

        def preprocess_for_lpips(np_img):
            # Convert to float32, resize to [H, W, 3] if needed
            if np_img.dtype != np.float32:
                np_img = np_img.astype(np.float32)
            if np_img.max() > 1.0:
                np_img = np_img / 255.0
            # LPIPS expects shape [1, 3, H, W] and range [-1, 1]
            if np_img.shape[-1] == 3:
                np_img = np.transpose(np_img, (2, 0, 1))  # [3, H, W]
            elif np_img.shape[0] == 3:
                pass  # already [3, H, W]
            else:
                raise ValueError(f"Unexpected image shape for LPIPS: {np_img.shape}")
            tensor = torch.from_numpy(np_img).unsqueeze(0)  # [1, 3, H, W]
            tensor = tensor * 2 - 1  # [0,1] -> [-1,1]
            return tensor
            
        target_bbox = artifact['target_bbox']
        orig_bbox = self._scale_bbox_to_image_size(target_bbox, orig_img.shape, img.shape) #  //TODO check if this is needed.

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        orig_crop = preprocess_for_lpips(self._crop_to_bbox(orig_img, orig_bbox)).to(device)
        img_crop = preprocess_for_lpips(self._crop_to_bbox(img, target_bbox)).to(device)

        with torch.no_grad():
            similarity = 1 - self.lpips_model(orig_crop, img_crop).item()

        logger.info(f"LPIPS similarity: {similarity:.4f}")

        # If similarity is low enough, distortion is present
        if similarity < thresholds['similar'] and similarity > thresholds['different']:
            passed = True
        else:
            passed = False

        return passed, similarity
    
    def _scale_bbox_to_image_size(self, bbox, target_img_shape, source_img_shape):
        """Scale bounding box coordinates from source image size to target image size."""
        xmin, ymin, xmax, ymax = bbox
        source_h, source_w = source_img_shape[:2]
        target_h, target_w = target_img_shape[:2]
        
        # Calculate scaling ratios
        scale_x = target_w / source_w
        scale_y = target_h / source_h
        
        # Scale coordinates
        scaled_xmin = int(round(xmin * scale_x))
        scaled_ymin = int(round(ymin * scale_y))
        scaled_xmax = int(round(xmax * scale_x))
        scaled_ymax = int(round(ymax * scale_y))
        
        # Ensure coordinates are within bounds
        scaled_xmin = max(0, scaled_xmin)
        scaled_ymin = max(0, scaled_ymin)
        scaled_xmax = min(target_w, scaled_xmax)
        scaled_ymax = min(target_h, scaled_ymax)
        
        return [scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax]

    def _crop_to_bbox(self, image: np.ndarray, bbox) -> np.ndarray:
        """Crop image to bounding box coordinates"""
        h, w = image.shape[:2]
        xmin, ymin, xmax, ymax = [int(round(x)) for x in bbox]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        return image[ymin:ymax, xmin:xmax]
    
    def mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert binary mask to bounding box [x_min, y_min, x_max, y_max]"""
        # Find coordinates where mask is non-zero
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Empty mask, return [0, 0, 0, 0]
            return [0, 0, 0, 0]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]
    
    # def save_lpips_bbox_figure(self, exp_id: str, artifact_img_array: np.ndarray, 
    #                           distortion_artifacts: List[Dict], similarity_scores: List[float], logger) -> None:
    #     """
    #     Save a figure showing bbox regions of distortion artifacts with LPIPS similarity scores as labels.
        
    #     Args:
    #         exp_id: Experiment identifier
    #         artifact_img_array: The artifact image as numpy array
    #         distortion_artifacts: List of distortion artifact dictionaries
    #         similarity_scores: List of corresponding LPIPS similarity scores
    #         logger: Logger instance
    #     """
    #     # Create output directory for LPIPS figures
    #     lpips_output_dir = self.output_dir / "lpips_figures"
    #     lpips_output_dir.mkdir(exist_ok=True)
        
    #     # Create figure and axis
    #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
    #     # Display the artifact image
    #     ax.imshow(artifact_img_array)
    #     ax.set_title(f'LPIPS Distortion Analysis - Experiment {exp_id}', fontsize=14, fontweight='bold')
    #     ax.axis('off')
        
    #     # Add bbox rectangles with similarity scores for each distortion artifact
    #     for i, (artifact, similarity_score) in enumerate(zip(distortion_artifacts, similarity_scores)):
    #         bbox = artifact['target_bbox']
    #         xmin, ymin, xmax, ymax = bbox
            
    #         # Create rectangle patch
    #         rect = patches.Rectangle(
    #             (xmin, ymin), 
    #             xmax - xmin, 
    #             ymax - ymin,
    #             linewidth=3, 
    #             edgecolor='red', 
    #             facecolor='none',
    #             alpha=0.8
    #         )
    #         ax.add_patch(rect)
            
    #         # Add similarity score label
    #         kernel_type = artifact.get('distortion_kernel', 'unknown')
    #         label_text = f'{kernel_type}\nLPIPS: {similarity_score:.3f}'
            
    #         # Position label on top of bbox with background
    #         ax.text(xmin + 5, ymin - 5, label_text, 
    #                fontsize=12, fontweight='bold', 
    #                color='white', 
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
    #                verticalalignment='bottom')
        
    #     # Add legend explaining the visualization
    #     legend_text = (
    #         'Red boxes: Distortion artifact regions\n'
    #         'LPIPS scores: Similarity between original and artifact regions\n'
    #         'Lower scores = more distortion detected'
    #     )
    #     ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
    #            fontsize=10, verticalalignment='top', 
    #            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
    #     # Save figure
    #     output_file = lpips_output_dir / f"lpips_analysis_{exp_id}.png"
    #     plt.tight_layout()
    #     plt.savefig(output_file, dpi=150, bbox_inches='tight')
    #     plt.close()
        
    #     logger.info(f"Saved LPIPS bbox figure for experiment {exp_id}: {output_file}")
    
    def filter_experiment_lpips(self, exp_id: str, logger) -> Dict:
        """
        Phase 1: Filter distortion artifacts in a single experiment using LPIPS.
        
        Args:
            exp_id: Experiment identifier
            
        Returns:
            Dict: Results with distortion artifact filtering status
        """
        data = self.experiment_data[exp_id]
        with open(data['metadata_path'], 'rb') as f:
            metadata = pickle.load(f)        
        
        artifact_image = data['artifact_path']
        real_image = data['real_image_path']
        original_img = Image.open(real_image)
        original_img_array = np.array(original_img.convert('RGB'))
        
        # Load artifact image
        artifact_img = Image.open(artifact_image)
        artifact_img_array = np.array(artifact_img.convert('RGB'))
        
        # Filter only distortion artifacts
        lpips_results = {}
        distortion_artifacts = []
        similarity_scores = []
        
        for artifact_idx, artifact in enumerate(metadata['artifacts']):
            artifact_type = artifact['artifact_type']
            
            if artifact_type == 'distortion':
                kernel_type = artifact['distortion_kernel']
                
                # Apply LPIPS filtering - now returns tuple (passed, similarity_score)
                lpips_passed, similarity_score = self.check_distortion_with_lpips(
                    artifact, artifact_type, original_img_array, artifact_img_array, exp_id, logger
                )
                
                # Collect distortion artifacts and scores for figure generation
                distortion_artifacts.append(artifact)
                similarity_scores.append(similarity_score)
                
                logger.info(f"Distortion artifact {artifact_idx} with {kernel_type}: {'PASSED' if lpips_passed else 'FAILED'} (LPIPS)")
                
                # Update kernel stats
                if kernel_type not in self.kernel_stats:
                    self.kernel_stats[kernel_type] = {'total': 0, 'passed': 0}
                self.kernel_stats[kernel_type]['total'] += 1
                if lpips_passed:
                    self.kernel_stats[kernel_type]['passed'] += 1
                
                lpips_results[artifact_idx] = lpips_passed
            else:
                # Non-distortion artifacts pass LPIPS phase (will be filtered in Phase 2)
                lpips_results[artifact_idx] = True
        
        # # Generate and save LPIPS bbox figure if there are distortion artifacts
        # if distortion_artifacts:
        #     self.save_lpips_bbox_figure(exp_id, artifact_img_array, distortion_artifacts, similarity_scores, logger)
        
        return lpips_results
    
    def process_experiment_explanations_worker(self, exp_id: str) -> Tuple[bool, float, Optional[str]]:
        """
        Worker function for Phase 2: Generate explanations for pre-filtered artifacts.
        
        Args:
            exp_id: Experiment identifier
            
        Returns:
            Tuple[bool, float, Optional[str], Optional[Dict]]: (success, cost, error_message, discarded_data)
        """
        # Create a separate money manager for this thread
        thread_money_manager = MoneyManager(model="gpt-4o")
        thread_client = OpenAI()
        
        # Create a thread-specific logger
        logger = logging.getLogger(__name__)
        data = self.experiment_data[exp_id]
        with open(data['metadata_path'], 'rb') as f:
            metadata = pickle.load(f)        
        
        artifact_image = data['artifact_path']
        real_image = data['real_image_path']
        original_img = Image.open(real_image)
        original_img_array = np.array(original_img.convert('RGB'))
        
        # Load artifact image
        artifact_img = Image.open(artifact_image)
        artifact_img_array = np.array(artifact_img.convert('RGB'))
        
        # Get LPIPS filter results for this experiment
        exp_lpips_results = self.lpips_filter_results.get(exp_id, {})
        
        # Process artifacts and generate explanations (ALL-OR-NOTHING approach)
        all_artifacts_passed = True
        processed_artifacts = []
        
        for artifact_idx, artifact in enumerate(metadata['artifacts']):
            artifact_type = artifact['artifact_type']
            target_mask = np.array(artifact['target_mask'], dtype=np.uint8) * 255
            
            # Check if artifact passed LPIPS filtering
            passed_lpips = exp_lpips_results.get(artifact_idx, True)  # Default to True for non-distortion
            
            if not passed_lpips:
                # Artifact was filtered out by LPIPS - entire experiment fails
                logger.info(f"Distortion artifact {artifact_idx}: FILTERED OUT by LPIPS - DISCARDING ENTIRE EXPERIMENT")
                discarded_data = {
                    'exp_id': exp_id,
                    'failure_reason': 'LPIPS_FILTER_FAILED',
                    'failed_artifact_idx': artifact_idx,
                    'failed_artifact_type': artifact_type,
                    'total_artifacts': len(metadata['artifacts']),
                    'processed_artifacts': len(processed_artifacts),
                    'metadata': metadata,
                    'artifact_image_path': str(artifact_image)
                }
                return False, thread_money_manager.total_cost, None, discarded_data
            
            # Create binary mask for image processing
            binary_mask = target_mask > 0  # True where mask is non-zero
            
            # Create the three required images for artifact_description using cropping and resizing
            # Find bounding box of the target region
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Crop the original and artifact images to the bounding box
            original_cropped = original_img_array[y_min:y_max+1, x_min:x_max+1]
            artifact_cropped = artifact_img_array[y_min:y_max+1, x_min:x_max+1]
            
            # Get the longest side of the original image
            orig_h, orig_w = original_img_array.shape[:2]
            max_orig_side = max(orig_h, orig_w)
            
            # Get dimensions of cropped region
            crop_h, crop_w = original_cropped.shape[:2]
            max_crop_side = max(crop_h, crop_w)
            
            # Calculate resize ratio
            resize_ratio = max_orig_side / max_crop_side
            new_h = int(crop_h * resize_ratio)
            new_w = int(crop_w * resize_ratio)
            
            # Function to convert to PIL for resizing
            def to_pil_for_resize(img):
                arr = img
                if arr.dtype in (np.float32, np.float64):
                    if arr.max() <= 1.0:
                        arr = (arr * 255).clip(0, 255)
                    arr = arr.astype(np.uint8)
                if arr.ndim == 2:
                    return Image.fromarray(arr)
                elif arr.shape[2] == 1:
                    return Image.fromarray(arr[:, :, 0])
                elif arr.shape[2] == 3:
                    return Image.fromarray(arr)
                elif arr.shape[2] == 4:
                    return Image.fromarray(arr[:, :, :3])
                else:
                    raise ValueError(f"Unexpected image shape: {arr.shape}")
            
            # Convert to PIL images for resizing
            original_pil = to_pil_for_resize(original_cropped)
            artifact_pil = to_pil_for_resize(artifact_cropped)
            
            # Resize
            original_resized = original_pil.resize((new_w, new_h), Image.LANCZOS)
            artifact_resized = artifact_pil.resize((new_w, new_h), Image.LANCZOS)
            
            # Convert back to numpy arrays
            target_original_img_array = np.array(original_resized)
            target_artifact_img_array = np.array(artifact_resized)
            
            # Convert back to original data type if needed
            if original_img_array.dtype in (np.float32, np.float64):
                target_original_img_array = target_original_img_array.astype(original_img_array.dtype) / 255.0
                target_artifact_img_array = target_artifact_img_array.astype(original_img_array.dtype) / 255.0
            
            # 1. Original image with target region masked out (filled with black)
            masked_original_img_array = original_img_array.copy()
            masked_original_img_array[binary_mask] = 0
            
            # Create object name description
            entity = artifact['entity']
            if artifact_type in ['addition', 'removal', 'distortion']:
                if artifact_type == 'distortion':
                    object_name = f"a {entity}"
                else:
                    subentity = artifact['subentity']
                    object_name = f"a {subentity} of a {entity}"
            elif artifact_type == 'fusion':
                fused_entity = artifact['fused_entity']
                object_name = f"a {entity} and a {fused_entity}"

            # Generate explanation for all artifacts
            try:
                if artifact_type in ['addition', 'removal', 'fusion']:
                    # Use artifact_description for both filtering and explanation
                    result = artifact_description(
                        thread_client, 
                        masked_original_img_array, 
                        target_original_img_array, 
                        target_artifact_img_array, 
                        object_name, 
                        artifact_type, 
                        thread_money_manager
                    )
                    
                    passed = result.has_artifact
                    explanation = result.explanation
                    label = result.label
                    
                    logger.info(f"{artifact_type} artifact {artifact_idx}: {'PASSED' if passed else 'FAILED'}")
                    logger.info(f"  Explanation: {explanation}")
                    logger.info(f"  Label: {label}")
                else:  # distortion - already passed LPIPS, just generate explanation
                    kernel_type = artifact['distortion_kernel']
                    
                    # Generate explanation using artifact_description
                    result = artifact_description(
                        thread_client, 
                        masked_original_img_array, 
                        target_original_img_array, 
                        target_artifact_img_array, 
                        object_name, 
                        artifact_type, 
                        thread_money_manager
                    )
                    explanation = result.explanation
                    label = result.label
                    passed = True  # Already passed LPIPS test
                    
                    logger.info(f"Distortion artifact {artifact_idx} with {kernel_type}: PASSED (pre-filtered by LPIPS)")
                    logger.info(f"  Explanation: {explanation}")
                    logger.info(f"  Label: {label}")
                
                # If this artifact failed, entire experiment fails
                if not passed:
                    logger.info(f"Artifact {artifact_idx} failed - DISCARDING ENTIRE EXPERIMENT {exp_id}")
                    discarded_data = {
                        'exp_id': exp_id,
                        'failure_reason': 'GPT4_FILTER_FAILED',
                        'failed_artifact_idx': artifact_idx,
                        'failed_artifact_type': artifact_type,
                        'total_artifacts': len(metadata['artifacts']),
                        'processed_artifacts': len(processed_artifacts),
                        'metadata': metadata,
                        'artifact_image_path': str(artifact_image),
                        'explanation_attempt': explanation if 'explanation' in locals() else "",
                        'label_attempt': label if 'label' in locals() else ""
                    }
                    return False, thread_money_manager.total_cost, None, discarded_data
                
                # Add explanation to artifact metadata
                artifact['explanation'] = explanation
                artifact['label'] = label
                processed_artifacts.append(artifact)
                
            except Exception as e:
                logger.error(f"Error processing artifact {artifact_idx} in experiment {exp_id}: {str(e)}")
                discarded_data = {
                    'exp_id': exp_id,
                    'failure_reason': 'PROCESSING_ERROR',
                    'failed_artifact_idx': artifact_idx,
                    'failed_artifact_type': artifact_type,
                    'total_artifacts': len(metadata['artifacts']),
                    'processed_artifacts': len(processed_artifacts),
                    'metadata': metadata,
                    'artifact_image_path': str(artifact_image),
                    'error_message': str(e)
                }
                return False, thread_money_manager.total_cost, str(e), discarded_data
                
        # Only save experiment if ALL artifacts passed
        if all_artifacts_passed and processed_artifacts:
            # Update metadata with all processed artifacts
            metadata['artifacts'] = processed_artifacts

            artifact_list = [
                f"bbox:{artifact['target_bbox']} description:{artifact['explanation']}"
                for artifact in processed_artifacts
            ]

            caption = artifact_explanation(
                thread_client,
                original_img_array,
                artifact_img_array,
                artifact_list,
                thread_money_manager
            )
            
            metadata['caption'] = caption
            results = {
                'metadata': metadata,
                'artifact_image': artifact_image,
            }


            
            with self.results_lock:
                self.filtered_results[exp_id] = results
            
            logger.info(f"Experiment {exp_id}: ALL {len(processed_artifacts)} artifacts PASSED - experiment saved")
            return True, thread_money_manager.total_cost, None, None
        else:
            logger.info(f"Experiment {exp_id}: DISCARDED due to artifact failure(s)")
            # This should not normally happen since we return early on failures
            discarded_data = {
                'exp_id': exp_id,
                'failure_reason': 'UNKNOWN_FAILURE',
                'failed_artifact_idx': -1,
                'failed_artifact_type': 'unknown',
                'total_artifacts': len(metadata['artifacts']),
                'processed_artifacts': len(processed_artifacts),
                'metadata': metadata,
                'artifact_image_path': str(artifact_image)
            }
            return False, thread_money_manager.total_cost, "Unknown failure", discarded_data

    def save_single_experiment(self, exp_id: str, results: Dict) -> None:
        """Save a single filtered experiment with explanations to output directory"""
        print(f"Saving experiment {exp_id}...")
        
        # Create experiment directory in output
        exp_output_dir = self.output_dir / f"{exp_id}"
        exp_output_dir.mkdir(exist_ok=True)
        
        # Get metadata
        metadata = results['metadata']
        
        # 1. Copy original image
        real_image = self.experiment_data[exp_id]['real_image_path']
        shutil.copy2(real_image, exp_output_dir / "real_image.png")
        print(f"  Copied original image: {real_image}")
        
        # 2. Copy artifact image
        artifact_image = results['artifact_image']
        shutil.copy2(artifact_image, exp_output_dir / "artifact_image.png")
        print(f"  Copied artifact image: {artifact_image}")

        # 3. Copy comparison image if it exists
        comparison_image = self.experiment_data[exp_id]['flux_dir'] / 'comparison.png'
        if comparison_image.exists():
            shutil.copy2(comparison_image, exp_output_dir / "comparison.png")
            print(f"  Copied comparison image: {comparison_image}")
        
        # 4. Create metadata dictionary with explanations
        metadata_dict = {}
        artifact_list = []
        for artifact in metadata['artifacts']:
            metadata_entry = {
                "target_bbox": artifact['target_bbox'],
                "artifact_type": artifact['artifact_type'],
                "entity": artifact['entity'],
                "artifact_entity": artifact['fused_entity'] if artifact['artifact_type'] == 'fusion' else artifact['subentity'],
                "explanation": artifact['explanation'],  # Include explanation
                "label": artifact['label']
            }
            
            # Add distortion kernel if applicable
            if artifact['artifact_type'] == 'distortion':
                metadata_entry["distortion_kernel"] = artifact['distortion_kernel']
            
            artifact_list.append(metadata_entry)
        metadata_dict['artifacts'] = artifact_list
        metadata_dict['caption'] = metadata['caption']
        
        # Save JSON file with explanations
        json_file_path = exp_output_dir / "metadata.json"
        with open(json_file_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"Saved filtered data with explanations for experiment {exp_id}")

    def save_discarded_experiments(self):
        """Save all discarded experiments to a JSON file"""
        if not self.discarded_experiments:
            print("No discarded experiments to save.")
            return
            
        discarded_file = self.output_dir / "discarded_experiments.json"
        
        # Create summary statistics
        failure_reasons = {}
        for exp_data in self.discarded_experiments.values():
            reason = exp_data['failure_reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        # Prepare output data
        output_data = {
            'summary': {
                'total_discarded': len(self.discarded_experiments),
                'failure_reasons': failure_reasons,
                'timestamp': datetime.now().isoformat()
            },
            'discarded_experiments': self.discarded_experiments
        }
        
        with open(discarded_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Saved {len(self.discarded_experiments)} discarded experiments to: {discarded_file}")
        
        # Print summary
        print("\nDiscarded Experiments Summary:")
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} experiments")

    def update_summary_file(self, exp_id: str, results: Dict, passed_experiments: int, total_experiments: int):
        """Update the summary file with results from a single experiment (thread-safe)"""
        summary_file = self.output_dir / "filtering_summary.txt"    
        
        # Create or append to summary file (thread-safe)
        mode = 'w' if passed_experiments == 1 else 'a'
        with open(summary_file, mode) as f:
            if passed_experiments == 1:  # First experiment, write header
                f.write("Unified Data Pipeline Summary (ALL-OR-NOTHING Mode)\n")
                f.write("=" * 60 + "\n")
                f.write("Note: Experiments are discarded if ANY artifact fails\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total experiments to process: {total_experiments}\n\n")
            
            f.write(f"Experiment {exp_id}:\n")
            f.write(f"  - Artifact image: {results['artifact_image'].name}\n")
            f.write(f"  - Artifacts passed: {len(results['metadata']['artifacts'])}\n")
            
            # List each artifact with its explanation
            for i, artifact in enumerate(results['metadata']['artifacts']):
                f.write(f"  - Artifact {i+1}: {artifact['artifact_type']} - {artifact['explanation'][:100]}... {artifact['label']}\n")
            f.write("\n")
    
    def run_pipeline(self):
        """Run the complete unified data pipeline in two sequential phases"""
        logger = setup_logging(self.output_dir)
        logger.info(f"Starting unified pipeline for {self.flux_dir}")
        
        # Step 1: Find matching experiments
        print("\n1. Finding matching experiments...")
        self.find_matching_experiments()
        
        if not self.experiment_data:
            print("No matching experiments found!")
            return
        
        total_experiments = len(self.experiment_data)
        
        # PHASE 1: LPIPS Filtering for Distortion Artifacts
        print("\n" + "=" * 50)
        print("PHASE 1: LPIPS DISTORTION FILTERING")
        print("=" * 50)
        
        total_distortion_artifacts = 0
        passed_distortion_artifacts = 0
        
        for exp_id in self.experiment_data.keys():
            print(f"\nFiltering experiment {exp_id} with LPIPS...")
            
            # Filter distortion artifacts using LPIPS
            lpips_results = self.filter_experiment_lpips(exp_id, logger)
            self.lpips_filter_results[exp_id] = lpips_results
            
            # Count distortion artifacts
            distortion_count = sum(1 for idx, result in lpips_results.items() 
                                 if idx < len(self.get_experiment_metadata(exp_id)['artifacts']) 
                                 and self.get_experiment_metadata(exp_id)['artifacts'][idx]['artifact_type'] == 'distortion')
            passed_distortion_count = sum(1 for idx, result in lpips_results.items() 
                                        if idx < len(self.get_experiment_metadata(exp_id)['artifacts']) 
                                        and self.get_experiment_metadata(exp_id)['artifacts'][idx]['artifact_type'] == 'distortion' 
                                        and result)
            
            total_distortion_artifacts += distortion_count
            passed_distortion_artifacts += passed_distortion_count
            
            print(f"Experiment {exp_id}: {passed_distortion_count}/{distortion_count} distortion artifacts passed LPIPS")
        
        print(f"\nPhase 1 complete: {passed_distortion_artifacts}/{total_distortion_artifacts} distortion artifacts passed LPIPS")
        
        # Print kernel stats for distortion
        if self.kernel_stats:
            print("\nDistortion kernel statistics:")
            for kernel, stats in self.kernel_stats.items():
                pass_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"  {kernel}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
        
        # PHASE 2: Explanation Generation for All Passing Artifacts (Multithreaded)
        print("\n" + "=" * 50)
        print("PHASE 2: EXPLANATION GENERATION (MULTITHREADED)")
        print("=" * 50)
        
        passed_experiments = 0
        total_phase2_cost = 0.0
        phase2_errors = 0
        
        # Use ThreadPoolExecutor for Phase 2
        max_workers = min(getattr(self, 'max_workers', 8), len(self.experiment_data))  # Don't create more threads than experiments
        print(f"Using {max_workers} threads for explanation generation...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all explanation generation tasks
            futures = {
                executor.submit(self.process_experiment_explanations_worker, exp_id): exp_id
                for exp_id in self.experiment_data.keys()
            }
            
            # Initialize progress bar for Phase 2
            progress_bar = tqdm(
                total=len(futures),
                desc="Phase 2 Progress", 
                unit="exp",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
            
            # Process results as they complete
            for future in as_completed(futures):
                exp_id = futures[future]
                try:
                    success, cost, error, discarded_data = future.result()
                    total_phase2_cost += cost
                    
                    # Store discarded experiment data if applicable
                    if discarded_data:
                        with self.discarded_lock:
                            self.discarded_experiments[exp_id] = discarded_data
                    
                    if success:
                        passed_experiments += 1
                        
                        # Get the results for this experiment (thread-safe)
                        with self.results_lock:
                            if exp_id in self.filtered_results:
                                results = self.filtered_results[exp_id]
                                
                                # Save this experiment immediately
                                self.save_single_experiment(exp_id, results)
                                
                                # Update summary file (thread-safe)
                                with self.summary_lock:
                                    self.update_summary_file(exp_id, results, passed_experiments, total_experiments)
                                
                                # Clear this experiment from memory to save space
                                del self.filtered_results[exp_id]
                    
                    # Update progress bar with current statistics
                    discarded_count = len(self.discarded_experiments)
                    progress_bar.set_postfix({
                        '✓': passed_experiments, 
                        '✗': discarded_count, 
                        'errors': phase2_errors
                    })
                    progress_bar.update(1)
                        
                except Exception as e:
                    phase2_errors += 1
                    error_msg = str(e)
                    logger.error(f"Error processing experiment {exp_id}: {error_msg}")
                    
                    # Update progress bar with error statistics
                    discarded_count = len(self.discarded_experiments)
                    progress_bar.set_postfix({
                        '✓': passed_experiments, 
                        '✗': discarded_count, 
                        'errors': phase2_errors
                    })
                    progress_bar.update(1)
            
            # Close progress bar when done
            progress_bar.close()
        
        # Update total cost with Phase 2 costs
        self.money_manager.total_cost += total_phase2_cost

        # Step 3: Final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY (ALL-OR-NOTHING MODE)")
        print("=" * 80)
        discarded_count = len(self.discarded_experiments)
        print(f"Phase 1 (LPIPS): {passed_distortion_artifacts}/{total_distortion_artifacts} distortion artifacts passed")
        print(f"Phase 2 (Explanations): {passed_experiments}/{total_experiments} experiments passed overall")
        print(f"Discarded experiments: {discarded_count}")
        print(f"Note: Experiments discarded if ANY artifact fails (all-or-nothing approach)")
        if phase2_errors > 0:
            print(f"Phase 2 errors: {phase2_errors}")
        print(f"Phase 2 threads: {max_workers}")
        print(f"Total cost: ${self.money_manager.total_cost:.4f}")
        
        # Update final summary in the file
        summary_file = self.output_dir / "filtering_summary.txt"
        with open(summary_file, 'a') as f:
            f.write(f"\nFinal Summary (ALL-OR-NOTHING Mode):\n")
            f.write(f"Phase 1 (LPIPS): {passed_distortion_artifacts}/{total_distortion_artifacts} distortion artifacts passed\n")
            f.write(f"Phase 2 (Overall): {passed_experiments}/{total_experiments} experiments passed filtering\n")
            f.write(f"Discarded experiments: {discarded_count}\n")
            f.write(f"Note: Experiments discarded if ANY artifact fails (all-or-nothing approach)\n")
            if phase2_errors > 0:
                f.write(f"Phase 2 errors: {phase2_errors}\n")
            f.write(f"Phase 2 threads used: {max_workers}\n")
            f.write(f"Total cost: ${self.money_manager.total_cost:.4f}\n")
            
            if self.kernel_stats:
                f.write("\nDistortion kernel statistics:\n")
                for kernel, stats in self.kernel_stats.items():
                    pass_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
                    f.write(f"  {kernel}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)\n")
        
        # Save discarded experiments
        print("\n" + "=" * 50)
        print("SAVING DISCARDED EXPERIMENTS")
        print("=" * 50)
        self.save_discarded_experiments()
        
        print(f"\nUnified pipeline complete! Check {self.output_dir} for results.")
    
    def get_experiment_metadata(self, exp_id: str) -> Dict:
        """Helper method to get experiment metadata"""
        data = self.experiment_data[exp_id]
        with open(data['metadata_path'], 'rb') as f:
            return pickle.load(f)


def main():
    """Main function to run the unified data pipeline"""
    parser = argparse.ArgumentParser(description='Unified pipeline for filtering and explaining GSAM and FLUX datasets')
    parser.add_argument('--gsam_dir', required=True, help='Directory containing GSAM generated data')
    parser.add_argument('--flux_dir', required=True, help='Directory containing FLUX generated data')
    parser.add_argument('--output_dir', required=True, help='Directory to save filtered datasets with explanations')
    parser.add_argument('--max_workers', type=int, default=64, help='Maximum number of threads for Phase 2 (default: 8)')
    
    args = parser.parse_args()
            
    gsam_dir = args.gsam_dir
    flux_dir = args.flux_dir
    output_dir = args.output_dir
    max_workers = args.max_workers
    
    print("Unified Data Pipeline - Sequential Filter + Multithreaded Explain (ALL-OR-NOTHING)")
    print("=" * 80)
    print(f"GSAM directory: {gsam_dir}")
    print(f"FLUX directory: {flux_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max workers for Phase 2: {max_workers}")
    print()
    
    # Initialize the pipeline
    print("Initializing unified pipeline...")
    pipeline = UnifiedDataPipeline(gsam_dir, flux_dir, output_dir)
    pipeline.max_workers = max_workers  # Set max workers
    
    # Run the complete pipeline
    pipeline.run_pipeline()
    
    print("\n" + "=" * 50)
    print("Unified pipeline completed successfully!")
    print(f"Check the output directory: {output_dir}")


if __name__ == "__main__":
    main()
