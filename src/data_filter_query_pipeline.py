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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from PIL import Image
import cv2

# Import query functions from prompts
from pipeline.prompts import query_addition_artifact_success, query_removal_artifact_success

# OpenAI client
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Please install with: pip install openai")
    sys.exit(1)


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
        
        self.experiment_data = {}
        self.filtered_results = {}

    
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
    
    def process_experiment(self, exp_id: str) -> bool:
        """
        Process a single experiment and determine if it passes the filtering criteria.
        
        Args:
            exp_id: Experiment identifier
            
        Returns:
            bool: True if experiment passes filtering criteria
        """

        data = self.experiment_data[exp_id]
        metadata = self.load_pickle_safely(data['gsam_files']['pickles'][0])
        artifact_type = list(metadata['artifacts'].keys())[0]

        # Find artifact images in FLUX data
        
        artifact_images = [img for img in data['flux_files']['images'] 
                            if f'artifact_{artifact_type}' in img.name.lower()]
        
        # Process each artifact image
        passed_images = []
        
        for img_path in artifact_images:
            # Load image
            img_array = self.load_image_safely(img_path)
            # Run GSAM detection
            target_mask =np.array(metadata['artifacts'][artifact_type]['annotation']['target_mask'], dtype=np.uint8)*255
            mask_image = Image.fromarray(target_mask)
            
            # Create part entity name description
            class_name = metadata['artifacts'][artifact_type]['class_name']
            main_entity = metadata['vocab'][0] if metadata['vocab'] else "object"
            part_entity_name = f"a {class_name} of a {main_entity}"
            print(artifact_type)
            # Apply appropriate filtering function based on artifact type
            if artifact_type == 'addition':
                # Check if the part is present in the mask region
                result = query_addition_artifact_success(
                    self.client, img_array, mask_image, part_entity_name
                )
                passed = result.get('success', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                print(f"Addition artifact {img_path.name}: {'PASSED' if passed else 'FAILED'}")
                print(f"  Reasoning: {reasoning}")
                
            else:  # removal
                # Check if the part is absent from the mask region
                result = query_removal_artifact_success(
                    self.client, img_array, mask_image, part_entity_name
                )
                passed = result.get('success', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                print(f"Removal artifact {img_path.name}: {'PASSED' if passed else 'FAILED'}")
                print(f"  Reasoning: {reasoning}")
            
            print(result)
            if passed:
                passed_images.append({
                    'path': img_path,
                    'reasoning': reasoning
                })
        
        # Experiment passes if at least one image passes
        experiment_passed = len(passed_images) > 0
        
        if experiment_passed:
            self.filtered_results[exp_id] = {
                'passed_images': passed_images,
                'metadata': metadata,
                'total_images': len(artifact_images),
                'passed_count': len(passed_images),
                'artifact_type': artifact_type
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
        metadata = results['metadata']
        artifact_types = list(metadata['artifacts'].keys())
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
        
        for img_data in results['passed_images']:
            img_path = img_data['path']
            if img_path.name == artifact_pattern:
                shutil.copy2(img_path, exp_output_dir / img_path.name)
                print(f"  Copied FLUX file: {img_path.name}")
        
        # Save query results for this experiment
        query_results_file = exp_output_dir / "query_results.txt"
        with open(query_results_file, 'w') as f:
            f.write(f"Query Results for Experiment {exp_id}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Artifact Type: {artifact_type}\n")
            f.write(f"Total Images: {results['total_images']}\n")
            f.write(f"Passed Images: {results['passed_count']}\n\n")
            
            for img_data in results['passed_images']:
                f.write(f"Image: {img_data['path'].name}\n")
                f.write(f"  Reasoning: {img_data['reasoning']}\n\n")
        
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
            f.write(f"  - Artifact type: {results['artifact_type']}\n")
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
                f.write(f"  - Artifact type: {results['artifact_type']}\n")
                f.write(f"  - Images passed: {results['passed_count']}/{results['total_images']}\n")
                f.write(f"  - Passed images: {[img.name for img in results['passed_images']]}\n\n")
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
            
            for img_data in results['passed_images']:
                img_path = img_data['path']
                if img_path.name == artifact_pattern:
                    shutil.copy2(img_path, exp_output_dir / img_path.name)
                    print(f"  Copied FLUX file: {img_path.name}")
            
            print(f"Saved filtered data for experiment {exp_id}")
        
        print(f"Filtering complete. Results saved to {self.output_dir}")
    
    def run_pipeline(self):
        """Run the complete data filtering pipeline"""
        print("Starting data filtering pipeline...")
        
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
            try:
                print(f"\nProcessing experiment {exp_id}...")
                
                # Process the experiment
                if self.process_experiment(exp_id):
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
                    
            except Exception as e:
                print(f"Error processing experiment {exp_id}: {e}")
                print(f"Skipping experiment {exp_id} and continuing with next...")
                continue
        
        # Step 3: Final summary
        print(f"\n3. Processing complete: {passed_experiments}/{total_experiments} experiments passed")
        
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