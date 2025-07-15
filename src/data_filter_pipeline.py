#!/usr/bin/env python3
"""
Data Filtering Pipeline for GSAM and FLUX Generated Datasets

This script processes GSAM and FLUX generated data, applies overlap/covering filters
based on artifact type, and saves filtered datasets to an output directory.

Usage:
    python data_filter_pipeline.py --gsam_dir <path> --flux_dir <path> --output_dir <path>
"""

import os
import sys
import glob
import pickle
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModel

import logging
from datetime import datetime
import torch.nn.functional as F

# Add pipeline to path for GSAM detector
sys.path.append('pipeline')
from pipeline.gsam_detector import GSAMDetector


def setup_logging(output_dir: str, supercategory: str):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'data_filtering_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class DataFilterPipeline:
    """Pipeline for filtering GSAM and FLUX generated datasets based on overlap/covering criteria"""
    
    def __init__(self, gsam_dir: str, flux_dir: str, output_dir: str):
        """
        Initialize the data filtering pipeline.
        
        Args:
            gsam_dir: Directory containing GSAM generated data
            flux_dir: Directory containing FLUX generated data  
            output_dir: Directory to save filtered datasets
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
        
        # Initialize GSAM detector
        self._init_gsam_detector()
        
        # Initialize DINO embedding
        self._init_dino_embedding()
        
        self.experiment_data = {}
        self.filtered_results = {}
        self.kernel_stats = {} 
        
    def _init_gsam_detector(self):
        """Initialize GSAM detector with pre-trained weights"""
        self.gsam_detector = GSAMDetector(
            grounding_checkpoint="weight/groundingdino_swint_ogc.pth",
            sam_checkpoint="weight/sam_vit_h_4b8939.pth",    
            sam_hq_checkpoint="weight/sam_hq_vit_h.pth",
            use_sam_hq=True,
            box_threshold=0.3,
            text_threshold=0.25,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("GSAM detector initialized successfully")

    def _init_dino_embedding(self):
        """Initialize DINOv2 embedding with pre-trained weights"""
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
        self.patch_size = self.dino_model.config.patch_size

        self.dino_model.eval()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dino_model.to(device)
        print("DINOv2 model initialized successfully")
    
    def find_matching_experiments(self) -> Dict[str, Dict]:
        """Find matching directories that start with 'image' between GSAM and FLUX experiments"""
        # Only consider directories that start with "image"
        gsam_image_dirs = {d.name: d for d in self.gsam_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('image')}
        flux_image_dirs = {d.name: d for d in self.flux_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('image')}
        
        # Find exact matches between image directories
        matching_ids = set(gsam_image_dirs.keys()) & set(flux_image_dirs.keys())
        
        print(f"GSAM image directories: {sorted(gsam_image_dirs.keys())}")
        print(f"FLUX image directories: {sorted(flux_image_dirs.keys())}")
        print(f"Found matching image directories: {sorted(matching_ids)}")
        
        self.experiment_data = {}
        
        # Process matching image directories
        for exp_id in matching_ids:
            # Get files from experiment directories
            gsam_files = self._get_files(gsam_image_dirs[exp_id])
            flux_files = self._get_files(flux_image_dirs[exp_id])
            
            # Find GSAM pickle files
            gsam_pickle_extensions = {'.pkl', '.pickle'}
            gsam_pickles = []
            for file_path in gsam_image_dirs[exp_id].rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in gsam_pickle_extensions:
                    gsam_pickles.append(file_path)
            gsam_files['pickles'] = gsam_pickles
            
            self.experiment_data[exp_id] = {
                'gsam_path': gsam_image_dirs[exp_id],
                'flux_path': flux_image_dirs[exp_id],
                'gsam_files': gsam_files,
                'flux_files': flux_files
            }
        
        total_matches = len(self.experiment_data)
        print(f"Found {total_matches} total matching experiments")
        print(f"Experiment IDs: {sorted(self.experiment_data.keys())}")
        
        return self.experiment_data
    
    def _get_files(self, directory: Path) -> Dict[str, List[Path]]:
        """Get all images and pickle files from a directory"""
        files = {'images': [], 'pickles': [], 'all_files': []}
        img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                files['all_files'].append(file_path)
                if file_path.suffix.lower() in img_extensions:
                    files['images'].append(file_path)
        
        return files
    
    def load_image_safely(self, image_path: Path) -> Optional[np.ndarray]:
        """Safely load an image file"""
        try:
            img = Image.open(image_path)
            return np.array(img.convert('RGB'))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_pickle_safely(self, pickle_path: Path) -> Optional[Any]:
        """Safely load a pickle file"""
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle {pickle_path}: {e}")
            return None
    
    def check_prediction_covers_bbox(self, gsam_metadata: Dict, artifact_type: str, 
                                           predictions: Dict, confidence_threshold: float = 0.3, 
                                           coverage_threshold: float = 0.4) -> bool:
        """
        Check if a prediction with target class covers more than the coverage threshold of target bbox.
        Used for addition type artifacts.
        """

        target_class_name = gsam_metadata['artifacts'][artifact_type]['class_name']
        vocab = gsam_metadata['vocab']
        target_class_idx = vocab.index(target_class_name)
        
        # Get target bbox
        target_bbox = gsam_metadata['artifacts'][artifact_type]['patch_data']['target_bbox']
        target_xmin, target_ymin, target_xmax, target_ymax = target_bbox
        
        # Check predictions
        for i, class_idx in enumerate(predictions['pred_classes']):
            if (class_idx.item() == target_class_idx and 
                predictions['scores'][i].item() >= confidence_threshold):
                
                # Get the prediction mask
                pred_mask = predictions['pred_masks'][i]
                
                # Create target bbox mask
                mask_height, mask_width = pred_mask.shape
                target_mask = torch.zeros((mask_height, mask_width), dtype=torch.bool)
                
                # Convert bbox coordinates to integer pixel coordinates
                bbox_xmin = max(0, int(target_xmin))
                bbox_ymin = max(0, int(target_ymin))
                bbox_xmax = min(mask_width, int(target_xmax))
                bbox_ymax = min(mask_height, int(target_ymax))
                
                # Fill target bbox area in mask
                target_mask[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = True
                
                # Calculate intersection and coverage ratio
                intersection_area = torch.sum(pred_mask & target_mask).item()
                target_bbox_area = torch.sum(target_mask).item()
                coverage_ratio = intersection_area / target_bbox_area if target_bbox_area > 0 else 0
                
                if coverage_ratio >= coverage_threshold:
                    return True
        
        return False

    
    def check_no_prediction_overlaps_bbox(self, gsam_metadata: Dict, artifact_type: str, 
                                         predictions: Dict, confidence_threshold: float = 0.3, 
                                         overlap_threshold: float = 0.1) -> bool:
        """
        Check if NO predictions with target class name overlap with the target bbox above a minimal threshold.
        Used for removal type artifacts.
        """

        target_class_name = gsam_metadata['artifacts'][artifact_type]['class_name']
        vocab = gsam_metadata['vocab']
        target_class_idx = vocab.index(target_class_name)
        
        # Get target bbox
        target_bbox = gsam_metadata['artifacts'][artifact_type]['patch_data']['target_bbox']
        target_xmin, target_ymin, target_xmax, target_ymax = target_bbox
        
        # Check predictions
        for i, class_idx in enumerate(predictions['pred_classes']):
            if (class_idx.item() == target_class_idx and 
                predictions['scores'][i].item() >= confidence_threshold):
                
                # Get the prediction mask
                pred_mask = predictions['pred_masks'][i]
                
                # Create target bbox mask
                mask_height, mask_width = pred_mask.shape
                target_mask = torch.zeros((mask_height, mask_width), dtype=torch.bool)
                
                # Convert bbox coordinates to integer pixel coordinates
                bbox_xmin = max(0, int(target_xmin))
                bbox_ymin = max(0, int(target_ymin))
                bbox_xmax = min(mask_width, int(target_xmax))
                bbox_ymax = min(mask_height, int(target_ymax))
                
                # Fill target bbox area in mask
                target_mask[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = True
                
                # Calculate intersection and overlap ratio
                intersection_area = torch.sum(pred_mask & target_mask).item()
                target_bbox_area = torch.sum(target_mask).item()
                overlap_ratio = intersection_area / target_bbox_area if target_bbox_area > 0 else 0
                
                # If ANY prediction overlaps above threshold, return False
                if overlap_ratio >= overlap_threshold:
                    return False
        
        # If no predictions overlap above threshold, return True
        return True

    def check_distortion_with_dino(self, gsam_metadata: Dict, artifact_type: str,
                                  orig_img: np.ndarray, img: np.ndarray, exp_id: str, logger) -> bool: 
        """
        Use DINO to check if two images are similar as a distortion artifact.
        Returns a dict with pass/fail, similarity, and classification.
        """
        # Set default thresholds
        thresholds = {
            'same': 0.9,      # Very high similarity
            'similar': 0.7,    # Moderate similarity
            'strange': 0.5,
            'different': 0.0   # Low similarity
        }

        mask = gsam_metadata['artifacts'][artifact_type]['patch_data'].get('masks', None)
        
        # Extract embeddings
        print("Extracting embeddings...")
        if mask is not None:
            print(f"Using mask for patch similarity: {mask.shape}")
            
        cls1, patches1 = self._extract_embeddings(orig_img, mask)
        cls2, patches2 = self._extract_embeddings(img, mask)
        
        # Compute similarities
        cls_similarity = self.compute_cosine_similarity(cls1, cls2)
        
        # Flatten patch embeddings for comparison
        patches1_flat = patches1.flatten(0, 1)  # [num_patches, hidden_size]
        patches2_flat = patches2.flatten(0, 1)  # [num_patches, hidden_size]
        
        # Use minimum number of patches for comparison
        min_patches = min(patches1_flat.shape[0], patches2_flat.shape[0])
        patches1_flat = patches1_flat[:min_patches]
        patches2_flat = patches2_flat[:min_patches]
        
        patch_similarity = self.compute_cosine_similarity(patches1_flat, patches2_flat)
        
        # Use average of CLS and patch similarities
        avg_similarity = (cls_similarity + patch_similarity) / 2
        
        logger.info(f"CLS similarity: {cls_similarity:.3f}, Patch similarity: {patch_similarity:.3f}, Avg: {avg_similarity:.3f}")

        # if avg_similarity < thresholds['same'] and avg_similarity >= thresholds['similar']:
        if cls_similarity < thresholds['same'] and cls_similarity >= thresholds['similar'] and patch_similarity < 0.8 and patch_similarity >= thresholds['strange']:
            passed = True
        else:
            passed = False
        # else:
        #     passed = False 

        return passed
    
    def _extract_embeddings(self, image: Image.Image, mask: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DINO embeddings from image
        """
        # Process image
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inputs = self.dino_processor(images=image, return_tensors="pt").to(device)
        
        # Get image dimensions
        batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
        num_patches_height = img_height // self.patch_size
        num_patches_width = img_width // self.patch_size
        num_patches_flat = num_patches_height * num_patches_width
        
        # Extract features
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            last_hidden_states = outputs[0]
        
        # Separate CLS token and patch embeddings
        cls_token = last_hidden_states[:, 0, :]  # [1, hidden_size]
        patch_features = last_hidden_states[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))
        
        # If mask is provided, filter patches
        if mask is not None:
            # Resize mask to patch grid
            mask_resized = self._resize_mask_to_patches(mask, num_patches_height, num_patches_width)
            # Get masked patch indices
            masked_patches = self._get_masked_patches(patch_features.squeeze(0), mask_resized)
            return cls_token.squeeze(0), masked_patches
        
        return cls_token.squeeze(0), patch_features.squeeze(0)  # Remove batch dimension
    
    def _resize_mask_to_patches(self, mask: np.ndarray, num_patches_height: int, num_patches_width: int) -> np.ndarray:
        """Resize mask to patch grid dimensions"""
        from PIL import Image
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_resized = mask_pil.resize((num_patches_width, num_patches_height), Image.NEAREST)
        return np.array(mask_resized) > 0
    
    def _get_masked_patches(self, patch_features: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """Extract patches where mask is True"""
        # Convert mask to boolean tensor
        mask_tensor = torch.from_numpy(mask).bool()
        
        # Get indices where mask is True
        masked_indices = torch.where(mask_tensor.flatten())[0]
        
        # Extract masked patches
        patch_features_flat = patch_features.flatten(0, 1)  # [num_patches, hidden_size]
        masked_patches = patch_features_flat[masked_indices]
        
        return masked_patches
    
    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings using L2 norm"""
        return F.normalize(embeddings, p=2, dim=-1)
    
    def compute_cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two normalized embeddings"""
        # Ensure embeddings are normalized
        emb1_norm = self.normalize_embeddings(emb1)
        emb2_norm = self.normalize_embeddings(emb2)
        
        # Compute cosine similarity
        # If embeddings are 1D, compute directly
        if emb1_norm.dim() == 1 and emb2_norm.dim() == 1:
            similarity = F.cosine_similarity(emb1_norm.unsqueeze(0), emb2_norm.unsqueeze(0), dim=1)
        else:
            # For multi-dimensional embeddings, flatten and compute
            emb1_flat = emb1_norm.flatten()
            emb2_flat = emb2_norm.flatten()
            similarity = F.cosine_similarity(emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0), dim=1)
        
        return similarity.item()
    
    def process_experiment(self, exp_id: str, logger) -> bool:
        """
        Process a single experiment and determine if it passes the filtering criteria.
        
        Args:
            exp_id: Experiment identifier
            
        Returns:
            bool: True if experiment passes filtering criteria
        """

        data = self.experiment_data[exp_id]
        gsam_metadata = self.load_pickle_safely(data['gsam_files']['pickles'][0])
        artifact_type = list(gsam_metadata['artifacts'].keys())[0]
        if artifact_type == 'distortion':
            kernel_type = gsam_metadata['artifacts'][artifact_type].get('kernel_type')

        # Find artifact images in FLUX data
        artifact_images = [img for img in data['flux_files']['images'] 
                            if f'artifact_{artifact_type}' in img.name.lower()]
        
        # Process each artifact image
        passed_images = []
        
        for img_path in artifact_images:
            # Load image
            img_array = self.load_image_safely(img_path)
            
            # Apply appropriate filtering function
            if artifact_type == 'addition':
                # Run GSAM detection
                predictions, _ = self.gsam_detector.detect_parts(
                    image=img_array, 
                    vocab=gsam_metadata['vocab']
                )
                passed = self.check_prediction_covers_bbox(
                    gsam_metadata, artifact_type, predictions
                )
                print(f"Addition artifact {img_path.name}: {'PASSED' if passed else 'FAILED'} (coverage test)")
            elif artifact_type == 'removal':
                # Run GSAM detection
                predictions, _ = self.gsam_detector.detect_parts(
                    image=img_array, 
                    vocab=gsam_metadata['vocab']
                )
                passed = self.check_no_prediction_overlaps_bbox(
                    gsam_metadata, artifact_type, predictions
                )
                print(f"Removal artifact {img_path.name}: {'PASSED' if passed else 'FAILED'} (no overlap test)")
            else:   # distortion
                original_path = [img for img in data['flux_files']['images'] if '01_original_image' in img.name.lower()]
                for orig_path in original_path:
                    original_img = self.load_image_safely(orig_path)
                passed = self.check_distortion_with_dino(
                    gsam_metadata, artifact_type, original_img, img_array, exp_id, logger
                )
                logger.info(f"Distortion artifact {exp_id} with {kernel_type}: {'PASSED' if passed else 'FAILED'} (similarity test)")

                if kernel_type not in self.kernel_stats:
                    self.kernel_stats[kernel_type] = {'total': 0, 'passed': 0}
                self.kernel_stats[kernel_type]['total'] += 1
                if passed:
                    self.kernel_stats[kernel_type]['passed'] += 1
                    passed_images.append(img_path)
            # if passed:
            #     passed_images.append(img_path)
        
        # Experiment passes if at least one image passes
        experiment_passed = len(passed_images) > 0
        
        if experiment_passed:
            self.filtered_results[exp_id] = {
                'passed_images': passed_images,
                'gsam_metadata': gsam_metadata,
                'total_images': len(artifact_images),
                'passed_count': len(passed_images)
            }
        
        print(f"Experiment {exp_id}: {len(passed_images)}/{len(artifact_images)} images passed")
        return experiment_passed

    def save_single_experiment(self, exp_id: str, results: Dict) -> None:
        """Save a single filtered experiment to output directory"""
        print(f"Saving experiment {exp_id}...")
        
        # Create experiment directory in output
        exp_output_dir = self.output_dir / f"filtered_{exp_id}"
        exp_output_dir.mkdir(exist_ok=True)
        
        # Get artifact type from metadata
        gsam_metadata = results['gsam_metadata']
        artifact_types = list(gsam_metadata['artifacts'].keys())
        if not artifact_types:
            print(f"No artifact types found for experiment {exp_id}, skipping")
            return
        
        artifact_type = artifact_types[0]  # Take first artifact type
        
        # Copy only specific GSAM files: 04_comparison_{artifact_type}*
        # Note: metadata.pkl files are excluded
        comparison_pattern = f"04_comparison_{artifact_type}"
        gsam_path = self.experiment_data[exp_id]['gsam_path']
        
        # Search for comparison files in GSAM directory
        for file_path in gsam_path.rglob('*'):
            if file_path.is_file() and comparison_pattern in file_path.name:
                # Copy directly to experiment directory (flatten structure)
                shutil.copy2(file_path, exp_output_dir / file_path.name)
                print(f"  Copied GSAM file: {file_path.name}")
        
        # Copy filtered FLUX images directly to experiment directory
        # Only copy artifact_{artifact_type}.png files from passed images
        artifact_pattern = f"artifact_{artifact_type}.png"
        
        for img_path in results['passed_images']:
            if img_path.name == artifact_pattern:
                shutil.copy2(img_path, exp_output_dir / img_path.name)
                print(f"  Copied FLUX file: {img_path.name}")
        
        print(f"Saved filtered data for experiment {exp_id}")

    def update_summary_file(self, exp_id: str, results: Dict, passed_experiments: int, total_experiments: int):
        """Update the summary file with results from a single experiment"""
        summary_file = self.output_dir / "filtering_summary.txt"
        
        # Create or append to summary file
        mode = 'w' if passed_experiments == 1 else 'a'
        with open(summary_file, mode) as f:
            if passed_experiments == 1:  # First experiment, write header
                f.write("Data Filtering Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total experiments to process: {total_experiments}\n\n")
            
            f.write(f"Experiment {exp_id}:\n")
            f.write(f"  - Images passed: {results['passed_count']}/{results['total_images']}\n")
            f.write(f"  - Passed images: {[img.name for img in results['passed_images']]}\n\n")
            
            # Update final summary at the end
            if passed_experiments == len([e for e in self.experiment_data.keys() if self.filtered_results.get(e)]):
                f.write(f"Final Summary: {passed_experiments} experiments passed filtering\n")

    def save_filtered_data(self):
        """Save filtered datasets to output directory"""
        if not self.filtered_results:
            print("No filtered data to save")
            return
        
        print(f"Saving filtered data to {self.output_dir}")
        
        # Create summary file
        summary_file = self.output_dir / "filtering_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Data Filtering Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total experiments processed: {len(self.experiment_data)}\n")
            f.write(f"Experiments passed filtering: {len(self.filtered_results)}\n\n")
            
            for exp_id, results in self.filtered_results.items():
                f.write(f"Experiment {exp_id}:\n")
                f.write(f"  - Images passed: {results['passed_count']}/{results['total_images']}\n")
                f.write(f"  - Passed images: {[img.name for img in results['passed_images']]}\n\n")
        # Copy filtered datasets
        for exp_id, results in self.filtered_results.items():
            # Create experiment directory in output
            exp_output_dir = self.output_dir / f"filtered_{exp_id}"
            exp_output_dir.mkdir(exist_ok=True)
            
            # Get artifact type from metadata
            gsam_metadata = results['gsam_metadata']
            artifact_types = list(gsam_metadata['artifacts'].keys())
            if not artifact_types:
                print(f"No artifact types found for experiment {exp_id}, skipping")
                continue
            
            artifact_type = artifact_types[0]  # Take first artifact type
            
            # Copy only specific GSAM files: 04_comparison_{artifact_type}*
            # Note: metadata.pkl files are excluded
            comparison_pattern = f"04_comparison_{artifact_type}"
            gsam_path = self.experiment_data[exp_id]['gsam_path']
            
            # Search for comparison files in GSAM directory
            for file_path in gsam_path.rglob('*'):
                if file_path.is_file() and comparison_pattern in file_path.name:
                    # Copy directly to experiment directory (flatten structure)
                    shutil.copy2(file_path, exp_output_dir / file_path.name)
                    print(f"  Copied GSAM file: {file_path.name}")
            
            # Copy filtered FLUX images directly to experiment directory
            # Only copy artifact_{artifact_type}.png files from passed images
            artifact_pattern = f"artifact_{artifact_type}.png"
            
            for img_path in results['passed_images']:
                if img_path.name == artifact_pattern:
                    shutil.copy2(img_path, exp_output_dir / img_path.name)
                    print(f"  Copied FLUX file: {img_path.name}")
            
            print(f"Saved filtered data for experiment {exp_id}")
        
        print(f"Filtering complete. Results saved to {self.output_dir}")
    
    def run_pipeline(self):
        """Run the complete data filtering pipeline"""
        print("Starting data filtering pipeline...")

        logger = setup_logging(self.output_dir, f"")
        logger.info(f"Starting filtering pipeline for {self.flux_dir}")
        
            # Step 1: Find matching experiments
        print("\n1. Finding matching experiments...")
        self.find_matching_experiments()
        
        if not self.experiment_data:
            print("No matching experiments found!")
            return
        
        # Step 2: Process each experiment and save immediately
        print("\n2. Processing experiments...")
        total_experiments = len(self.experiment_data)
        passed_experiments = 0
        
        for exp_id in self.experiment_data.keys():
            # try:
            print(f"\nProcessing experiment {exp_id}...")
            
            # Process the experiment
            if self.process_experiment(exp_id, logger):
                passed_experiments += 1
                
                # Get the results for this experiment
                results = self.filtered_results[exp_id]
                
                # Save this experiment immediately
                self.save_single_experiment(exp_id, results)
                
                # Update summary file
                self.update_summary_file(exp_id, results, passed_experiments, total_experiments)
                
                # Clear this experiment from memory to save space
                del self.filtered_results[exp_id]
                print(f"Experiment {exp_id} saved and cleared from memory")
            else:
                print(f"Experiment {exp_id} did not pass filtering criteria")
                
            # except Exception as e:
            #     print(f"Error processing experiment {exp_id}: {e}")
            #     print(f"Skipping experiment {exp_id} and continuing with next...")
            #     continue
        
        # Step 3: Final summary
        print(f"\n3. Processing complete: {passed_experiments}/{total_experiments} experiments passed")

        # Update final summary in the file
        summary_file = self.output_dir / "filtering_summary.txt"
        with open(summary_file, 'a') as f:
            f.write("\nSuccess rate per kernel_type:")
            for ktype, stats in self.kernel_stats.items():
                total = stats['total']
                passed = stats['passed']
                rate = (passed / total) if total > 0 else 0.0
                f.write(f"  {ktype}: {passed}/{total} ({rate:.2%})")

        with open(summary_file, 'a') as f:
            f.write(f"\nFinal Summary: {passed_experiments}/{total_experiments} experiments passed filtering\n")
        
        print(f"\nPipeline complete! Check {self.output_dir} for results.")


def main():
    """Main function to run the data filtering pipeline"""
    parser = argparse.ArgumentParser(description='Filter GSAM and FLUX datasets based on overlap/covering criteria')
    parser.add_argument('--gsam_dir', help='Directory containing GSAM generated data')
    parser.add_argument('--flux_dir', help='Directory containing FLUX generated data')
    parser.add_argument('--output_dir', help='Directory to save filtered datasets')
    
    args = parser.parse_args()
            
    gsam_dir = args.gsam_dir
    flux_dir = args.flux_dir
    output_dir = args.output_dir
    
    print("Data Filtering Pipeline - Custom Configuration")
    print("=" * 50)
    print(f"GSAM directory: {gsam_dir}")
    print(f"FLUX directory: {flux_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize the pipeline
    print("Initializing pipeline...")
    pipeline = DataFilterPipeline(gsam_dir, flux_dir, output_dir)
    
    # Run the complete pipeline
    pipeline.run_pipeline()
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print(f"Check the output directory: {output_dir}")



if __name__ == "__main__":
    main() 