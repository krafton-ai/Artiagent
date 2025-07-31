#!/usr/bin/env python3
"""
Data Filtering Pipeline for GSAM and FLUX Generated Datasets

This script processes GSAM and FLUX generated data, applies GPT-4 Vision-based queries
to check artifact injection success, and saves filtered datasets to an output directory.

Usage:
    python data_filter_pipeline.py --gsam_dir <path> --flux_dir <path> --output_dir <path>
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModel
import logging
from datetime import datetime
import torch.nn.functional as F

# Import query functions from prompts
from pipeline.prompts import query_addition_artifact_success, query_removal_artifact_success, MoneyManager
from openai import OpenAI


def setup_logging(output_dir: str):
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
    """Pipeline for filtering GSAM and FLUX generated datasets using GPT-4 Vision queries"""
    
    def __init__(self, gsam_dir: str, flux_dir: str, output_dir: str):
        """
        Initialize the data filtering pipeline.
        
        Args:
            gsam_dir: Directory containing GSAM generated data
            flux_dir: Directory containing FLUX generated data  
            output_dir: Directory to save filtered datasets
            openai_api_key: OpenAI API key for GPT-4 Vision
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
        self._init_dino_embedding()
        self.money_manager = MoneyManager(model="gpt-4o")
        self.experiment_data = {}
        self.filtered_results = {}
        self.kernel_stats = {}

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
        for exp_id in matching_ids:
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

    def check_distortion_with_dino(self, metadata: Dict, artifact_type: str,
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

        mask = metadata['artifacts'][artifact_type]['patch_data'].get('masks', None)
        
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
        if cls_similarity < thresholds['same'] and patch_similarity < 0.8 and patch_similarity >= thresholds['strange']:
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
    
    def process_experiment(self, exp_id: str, logger) -> bool:
        """
        Process a single experiment and determine if it passes the filtering criteria.
        
        Args:
            exp_id: Experiment identifier
            
        Returns:
            bool: True if experiment passes filtering criteria
        """

        data = self.experiment_data[exp_id]
        with open(data['metadata_path'], 'rb') as f:
            metadata = pickle.load(f)        
        artifact_image = data['artifact_path']
        # Process each artifact image
        
        # Load image
        img = Image.open(artifact_image)
        img_array = np.array(img.convert('RGB'))
        # Run GSAM detection
        for artifact in metadata['artifacts']:
            artifact_type = artifact['artifact_type']

            target_mask =np.array(artifact['target_mask'], dtype=np.uint8)*255
            mask_image = Image.fromarray(target_mask)
            if artifact_type == 'distortion':
                kernel_type = artifact['distortion_kernel']
            # Create part entity name description
            subentity = artifact['subentity']
            entity = artifact['entity']
            part_entity_name = f"a {subentity} of a {entity}"
            # Apply appropriate filtering function based on artifact type
            if artifact_type == 'addition':
                # Check if the part is present in the mask region
                result = query_addition_artifact_success(
                    self.client, img_array, mask_image, part_entity_name, self.money_manager
                )
                passed = result.get('success', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                print(f"Addition artifact {artifact_image.name}: {'PASSED' if passed else 'FAILED'}")
                print(f"  Reasoning: {reasoning}")
                
            elif artifact_type == 'removal':  # removal
                # Check if the part is absent from the mask region
                result = query_removal_artifact_success(
                    self.client, img_array, mask_image, part_entity_name, self.money_manager
                )
                passed = result.get('success', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                print(f"Removal artifact {artifact_image.name}: {'PASSED' if passed else 'FAILED'}")
                print(f"  Reasoning: {reasoning}")
            else:  # distortion
                passed=True
                # real_image = data['real_image_path']
                # original_img = Image.open(real_image)
                # passed = self.check_distortion_with_dino(
                #     metadata, artifact_type, original_img, img_array, exp_id, logger
                # )
                # logger.info(f"Distortion artifact {exp_id} with {kernel_type}: {'PASSED' if passed else 'FAILED'} (similarity test)")

                # if kernel_type not in self.kernel_stats:
                #     self.kernel_stats[kernel_type] = {'total': 0, 'passed': 0}
                # self.kernel_stats[kernel_type]['total'] += 1
                # if passed:
                #     self.kernel_stats[kernel_type]['passed'] += 1

                
        if passed:
            self.filtered_results[exp_id] = {
                'metadata': metadata,
                'artifact_image': artifact_image
            }
        
        return passed

    def save_single_experiment(self, exp_id: str, results: Dict) -> None:
        """Save a single filtered experiment to output directory"""
        print(f"Saving experiment {exp_id}...")
        
        # Create experiment directory in output
        exp_output_dir = self.output_dir / f"filtered_{exp_id}"
        exp_output_dir.mkdir(exist_ok=True)
        
        # Get artifact type from metadata
        metadata = results['metadata']
        
        # 1. Find and copy original image
        real_image = self.experiment_data[exp_id]['real_image_path']
        shutil.copy2(real_image, exp_output_dir / "real_image.png")
        print(f"  Copied original image: {real_image}")
        
        # 2. Copy artifact image from passed images
        artifact_image = results['artifact_image']
        shutil.copy2(artifact_image, exp_output_dir / "artifact_image.png")
        print(f"  Copied artifact image: {artifact_image}")

        # 2. Copy comparison image from passed images
        comparison_image =  self.experiment_data[exp_id]['flux_dir'] / 'comparison.png'
        shutil.copy2(comparison_image, exp_output_dir / "comparison.png")
        print(f"  Copied comparison image: {comparison_image}")


        
        # Create metadata dictionary
        metadata_dict = []
        for artifact in metadata['artifacts']:
            metadata_dict.append({
                "target_bbox": artifact['target_bbox'],
                "artifact_type": artifact['artifact_type'],
                "entity": artifact['entity'],
                "subentity": artifact['subentity'],
                "distortion_kernel": artifact['distortion_kernel'] if artifact['artifact_type'] == 'distortion' else None
            })
        
        # Save JSON file
        json_file_path = exp_output_dir / "metadata.json"
        with open(json_file_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # 4. Create artifact image with bbox overlay
        # artifact_with_bbox_path = exp_output_dir / "artifact_image.png"
        # if artifact_with_bbox_path.exists():
        #     try:
        #         # Load the artifact image
        #         artifact_img = cv2.imread(str(artifact_with_bbox_path))
        #         if artifact_img is not None:
        #             # Draw bounding box rectangle
        #             x_min, y_min, x_max, y_max = target_bbox
        #             cv2.rectangle(artifact_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle, thickness 2
                    
        #             # Save the image with bbox overlay
        #             bbox_overlay_path = exp_output_dir / "artifact_with_bbox.png"
        #             cv2.imwrite(str(bbox_overlay_path), artifact_img)
        #             print(f"  Created artifact_with_bbox.png with target region highlighted")
        #         else:
        #             print(f"  Warning: Could not load artifact image for bbox overlay")
        #     except Exception as e:
        #         print(f"  Warning: Failed to create bbox overlay image: {e}")
        
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
            f.write(f"  - Images passed: {results['artifact_image']}\n")
            f.write(f"  - Passed images: {results['artifact_image']}\n\n")
            
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
                f.write(f"  - Passed images: {results['artifact_image']}\n\n")

        # Copy filtered datasets
        for exp_id, results in self.filtered_results.items():
            # Create experiment directory in output
            exp_output_dir = self.output_dir / f"filtered_{exp_id}"
            exp_output_dir.mkdir(exist_ok=True)
            
            # Get artifact type from metadata
            metadata = results['metadata']
            artifact_types = list(metadata['artifacts'].keys())
            if not artifact_types:
                print(f"No artifact types found for experiment {exp_id}, skipping")
                continue
            
            artifact_type = artifact_types[0]  # Take first artifact type
            real_image = self.experiment_data[exp_id]['real_image_path']
            if real_image:
                original_img_path = real_image
                shutil.copy2(original_img_path, exp_output_dir / "real_image.png")
                print(f"  Copied original image: {original_img_path.name}")
            else:
                print(f"  Warning: No original image found for experiment {exp_id}")
            
            img_path = results['artifact_image']
            shutil.copy2(img_path, exp_output_dir / "artifact_image.png")
            print(f"  Copied artifact image: {img_path.name}")
            
            # 3. Create JSON metadata file
            artifact_info = metadata['artifacts'][artifact_type]
            
            # Extract target mask and convert to bbox
            target_mask = np.array(artifact_info['annotation']['target_mask'], dtype=np.uint8)
            target_bbox = self.mask_to_bbox(target_mask)
            
            # Extract entity and part entity
            entity = metadata['vocab'][0] if metadata['vocab'] else "unknown"
            part_entity = artifact_info['class_name']
            
            # Create metadata dictionary
            metadata_dict = {
                "target_bbox": target_bbox,
                "artifact_type": artifact_type,
                "entity": entity,
                "part_entity": part_entity
            }
            
            # Save JSON file
            json_file_path = exp_output_dir / "metadata.json"
            with open(json_file_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # 4. Create artifact image with bbox overlay
            artifact_with_bbox_path = exp_output_dir / "artifact_image.png"
            if artifact_with_bbox_path.exists():
                try:
                    # Load the artifact image
                    artifact_img = cv2.imread(str(artifact_with_bbox_path))
                    if artifact_img is not None:
                        # Draw bounding box rectangle
                        x_min, y_min, x_max, y_max = target_bbox
                        cv2.rectangle(artifact_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle, thickness 2
                        
                        # Save the image with bbox overlay
                        bbox_overlay_path = exp_output_dir / "artifact_with_bbox.png"
                        cv2.imwrite(str(bbox_overlay_path), artifact_img)
                        print(f"  Created artifact_with_bbox.png with target region highlighted")
                    else:
                        print(f"  Warning: Could not load artifact image for bbox overlay")
                except Exception as e:
                    print(f"  Warning: Failed to create bbox overlay image: {e}")
            
            print(f"  Created metadata.json with bbox: {target_bbox}")
            print(f"Saved filtered data for experiment {exp_id}")
        
        print(f"Filtering complete. Results saved to {self.output_dir}")
    
    def run_pipeline(self):
        """Run the complete data filtering pipeline"""
        logger = setup_logging(self.output_dir)
        logger.info(f"Starting filtering pipeline for {self.flux_dir}")
        # print("Starting data filtering pipeline...")
        
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

        
        # Step 3: Final summary
        print(f"\n3. Processing complete: {passed_experiments}/{total_experiments} experiments passed")
        print(f"Total cost: {self.money_manager.total_cost}")
        
        # Update final summary in the file
        summary_file = self.output_dir / "filtering_summary.txt"
        with open(summary_file, 'a') as f:
            f.write(f"\nFinal Summary: {passed_experiments}/{total_experiments} experiments passed filtering\n")
        
        print(f"\nPipeline complete! Check {self.output_dir} for results.")


def main():
    """Main function to run the data filtering pipeline"""
    parser = argparse.ArgumentParser(description='Filter GSAM and FLUX datasets using GPT-4 Vision queries')
    parser.add_argument('--gsam_dir', required=True, help='Directory containing GSAM generated data')
    parser.add_argument('--flux_dir', required=True, help='Directory containing FLUX generated data')
    parser.add_argument('--output_dir', required=True, help='Directory to save filtered datasets')
    
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