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
from pipeline.prompts import artifact_success, MoneyManager
from openai import OpenAI
import lpips


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
        self._init_lpips()
        self.money_manager = MoneyManager(model="gpt-4o")
        self.experiment_data = {}
        self.filtered_results = {}
        self.kernel_stats = {}

    def _init_lpips(self):
        """Initialize LPIPS"""
        self.lpips_model = lpips.LPIPS(net='alex')
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.lpips_model.to(device)
        print("LPIPS initalized successfully")
    
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

    def check_distortion_with_lpips(self, artifact: Dict, artifact_type: str,
                                    orig_img: np.ndarray, img: np.ndarray, exp_id: str, logger) -> bool: 
        """
        Use DINO to check if two images are similar as a distortion artifact.
        Returns a dict with pass/fail, similarity, and classification.
        """
        # Set default thresholds
        thresholds = {
            'similar': 0.8,
            # 'strange': 0.7,
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
        orig_bbox = self._scale_bbox_to_image_size(target_bbox, orig_img.shape, img.shape)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        orig_crop = preprocess_for_lpips(self._crop_to_bbox(orig_img, orig_bbox)).to(device)
        img_crop = preprocess_for_lpips(self._crop_to_bbox(img, target_bbox)).to(device)

        with torch.no_grad():
            d = 1 - self.lpips_model(orig_crop, img_crop).item()

        logger.info(f"LPIPS similarity: {d:.4f}")

        if d < thresholds['similar']:
            passed = True
        else:
            passed = False

        return passed
    
    def _scale_bbox_to_image_size(self, bbox, target_img_shape, source_img_shape):
        """
        Scale bounding box coordinates from source image size to target image size.
        """
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
        real_image = data['real_image_path']
        original_img = Image.open(real_image)
        original_img_array = np.array(original_img.convert('RGB'))
        # Load image
        artifact_img = Image.open(artifact_image)
        artifact_img_array = np.array(artifact_img.convert('RGB'))
        
        # Filter artifacts - keep only those that pass the tests
        passing_artifacts = []
        
        # Run GSAM detection
        for artifact in metadata['artifacts']:
            artifact_type = artifact['artifact_type']

            target_mask =np.array(artifact['target_mask'], dtype=np.uint8)*255
            
            # Create binary mask for image processing
            binary_mask = target_mask > 0  # True where mask is non-zero
            
            # Create the three required images
            # 1. Original image with target region masked out (filled with black)
            masked_original_img_array = original_img_array.copy()
            masked_original_img_array[binary_mask] = 0
            
            # 2. Original image with only target region visible (everything else black)
            target_original_img_array = np.zeros_like(original_img_array)
            target_original_img_array[binary_mask] = original_img_array[binary_mask]
            
            # 3. Artifact image with only target region visible (everything else black)
            target_artifact_img_array = np.zeros_like(artifact_img_array)
            target_artifact_img_array[binary_mask] = artifact_img_array[binary_mask]
            
            if artifact_type == 'distortion':
                kernel_type = artifact['distortion_kernel']
            # Create part entity name description
            entity = artifact['entity']
            if artifact_type in ['addition', 'removal']:
                subentity = artifact['subentity']
                object_name = f"a {subentity} of a {entity}"
            elif artifact_type == 'fusion':
                fused_entity = artifact['fused_entity']
                object_name = f"a {entity} and a {fused_entity}"

            # Apply appropriate filtering function based on artifact type
            if artifact_type in ['addition', 'removal', 'fusion']:

                result = artifact_success(
                    self.client, masked_original_img_array, target_original_img_array, target_artifact_img_array, object_name, artifact_type, self.money_manager
                )
                passed = result.success
                reasoning = result.reasoning

                print(f"{artifact_type} artifact {artifact_image.name}: {'PASSED' if passed else 'FAILED'}")
                print(f"  Reasoning: {reasoning}")

            else:  # distortion
                passed = self.check_distortion_with_lpips(
                    artifact, artifact_type, original_img_array, artifact_img_array, exp_id, logger
                )
                logger.info(f"Distortion artifact with {kernel_type}: {'PASSED' if passed else 'FAILED'} (similarity test)")

                if kernel_type not in self.kernel_stats:
                    self.kernel_stats[kernel_type] = {'total': 0, 'passed': 0}
                self.kernel_stats[kernel_type]['total'] += 1
                if passed:
                    self.kernel_stats[kernel_type]['passed'] += 1
            
            # Only keep artifacts that passed the test
            if passed:
                passing_artifacts.append(artifact)
                
        # Update metadata with only passing artifacts
        metadata['artifacts'] = passing_artifacts
        
        # Only add experiment to filtered results if there are passing artifacts
        if passing_artifacts:
            self.filtered_results[exp_id] = {
                'metadata': metadata,
                'artifact_image': artifact_image
            }
        
        return len(passing_artifacts) > 0

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
                "artifact_entity": artifact['fused_entity'] if artifact['artifact_type'] == 'fusion' else artifact['subentity'],
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