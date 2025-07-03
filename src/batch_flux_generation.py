#!/usr/bin/env python3
"""
FLUX Artifact Generation Batch Processing Script

This script reads unified VLPart processing results and generates
artifacts using FLUX model for each annotation with patch annotations.

Features:
- Read unified VLPart processing data
- Generate artifacts for each annotation using FLUX with patch-based guidance
- Save visualizations and final results
- Progress tracking and resumption capability
"""

import os
import sys
import json
import time
import logging
import pickle
import glob
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import cv2
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from pipeline import FluxGenerator, FluxConfig, ImageVisualizer


def setup_logging(output_dir: str, supercategory: str):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'flux_generation_{supercategory}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_processed_data(file_path: str) -> Optional[Dict]:
    """Load unified processed data from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None


def create_flux_visualizations(img_array: np.ndarray, generated_image: np.ndarray,
                             annotation: Dict, artifact_type: str, 
                             patch_data: Dict, img_filename: str, 
                             caption: str, output_dir: str, visualizer: ImageVisualizer):
    """Create visualizations for FLUX generation results"""
    # Save comparison
    visualizer.show_comparison(
        img_array, generated_image, caption,
        titles=["Original", f"Generated ({artifact_type.title()})"],
        image_name=img_filename, 
        base_dir=output_dir,
        filename=f"04_comparison_{artifact_type}.png"
    )
    
    # Save patch annotation visualizations
    flux_output_dir = visualizer._create_output_dir(img_filename, output_dir)
    save_patch_visualizations(
        img_array, patch_data, artifact_type, img_filename, flux_output_dir
    )

    generated_image.save(flux_output_dir+'/'+f'07_injected_image_{artifact_type}.png')


def save_patch_visualizations(img_array: np.ndarray, patch_data: Dict,
                            artifact_type: str, img_filename: str, output_dir: str):
    """Save visualizations showing the patch annotations used"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    

    # Get patch indices with None handling
    reference_patches = patch_data.get('reference_patch_indices', []) or []
    target_patches = patch_data.get('target_patch_indices', []) or []
    
    # Convert patch indices (subtract 512 offset if present)
    reference_patches = [idx-512 for idx in reference_patches] if reference_patches else []
    target_patches = [idx-512 for idx in target_patches] if target_patches else []
    
    # Assume 16x16 patches for visualization (this could be made configurable)
    patch_size = 16
    h, w = img_array.shape[:2]
    patches_h = h // patch_size
    patches_w = w // patch_size
    # For addition artifacts, show 3 panels (original, reference patches, target patches)
    # For other artifacts, show 2 panels (original, reference patches only)
    if artifact_type == 'addition' and target_patches:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Reference patches overlay
        axes[1].imshow(img_array)
        for patch_idx in reference_patches:
            row = patch_idx // patches_w
            col = patch_idx % patches_w
            rect = patches.Rectangle((col * patch_size, row * patch_size), 
                                   patch_size, patch_size, 
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
            axes[1].add_patch(rect)
        axes[1].set_title(f'Reference Patches ({len(reference_patches)} patches)')
        axes[1].axis('off')
                # Target patches overlay

        axes[2].imshow(img_array)
        for patch_idx in target_patches:
            row = patch_idx // patches_w
            col = patch_idx % patches_w
            rect = patches.Rectangle((col * patch_size, row * patch_size), 
                                   patch_size, patch_size, 
                                   linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
            axes[2].add_patch(rect)
        axes[2].set_title(f'Target Patches ({len(target_patches)} patches)')
        axes[2].axis('off')
    else:
        # For removal and distortion, only show original and reference patches
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Reference patches overlay
        axes[1].imshow(img_array)
        for patch_idx in reference_patches:
            row = patch_idx // patches_w
            col = patch_idx % patches_w
            rect = patches.Rectangle((col * patch_size, row * patch_size), 
                                   patch_size, patch_size, 
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
            axes[1].add_patch(rect)
        axes[1].set_title(f'Reference Patches ({len(reference_patches)} patches)')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"06_patches_{artifact_type}.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()


def process_single_image(data_file: str, flux_generator: FluxGenerator,
                        visualizer: ImageVisualizer, artifact_types: List[str], 
                        output_dir: str, logger) -> Dict:
    """Process a single image with FLUX artifact generation"""
    # Load data
    data = load_processed_data(data_file)
    if data is None:
        return {'success': False, 'error': 'Failed to load data'}
    
    img_id = data['image_info']['id']
    img_filename = data['image_info']['file_name']
    
    logger.info(f"Processing FLUX generation for image {img_id}: {img_filename}")
    
    results = {
        'image_id': img_id,
        'filename': img_filename,
        'artifacts': {},
        'success': False,
        'error': None,
        'processing_time': 0
    }
    
    start_time = time.time()
    img_array = data['image_array']
    caption = data['caption']

    
    # Copy original image and detection results from VLPart output if needed
    segmentation_output_dir = os.path.dirname(os.path.dirname(data_file))  # Go up from processed_data
    flux_output_path = os.path.join(output_dir, img_filename)
    # os.makedirs(flux_output_path, exist_ok=True)
    
    # Copy visualizations from VLPart processing
    for viz_file in ["01_original_image.png", "02_detection_results.png"]:
        vlpart_viz_path = os.path.join(segmentation_output_dir, img_filename, viz_file)
        flux_viz_path = os.path.join(flux_output_path, viz_file)
        
        if os.path.exists(vlpart_viz_path) and not os.path.exists(flux_viz_path):
            import shutil
            shutil.copy2(vlpart_viz_path, flux_viz_path)
    
    # Process each artifact type
    successful_artifacts = 0
    
    for artifact_type in artifact_types:
        if artifact_type not in data['artifacts']:
            logger.warning(f"  No data found for {artifact_type}")
            continue
        
        artifact_data = data['artifacts'][artifact_type]
        
        # Check if artifact has error
        if 'error' in artifact_data:
            logger.error(f"  ❌ {artifact_type} has error: {artifact_data['error']}")
            results['artifacts'][artifact_type] = {
                'success': False,
                'error': artifact_data['error']
            }
            continue
        
        logger.info(f"  Generating {artifact_type} artifact with patch annotations...")
        
        annotation = artifact_data['annotation']
        class_name = artifact_data['class_name']
        patch_data = artifact_data['patch_data']
        
        # Validate patch annotations
        if not patch_data or 'error' in patch_data:
            logger.error(f"    No valid patch annotations found for {artifact_type}")
            results['artifacts'][artifact_type] = {
                'success': False,
                'error': 'No valid patch annotations available'
            }
            continue
        
        if not patch_data or 'error' in patch_data:
            print(f"    No valid patch annotations available for {artifact_type}")
            return None
        
        # Get patch indices with None handling
        reference_patch_indices = patch_data.get('reference_patch_indices', []) or []
        target_patch_indices = patch_data.get('target_patch_indices', []) or []
        
        # Log patch annotation usage
        ref_patches = len(reference_patch_indices)
        print(f"    Using {ref_patches} reference patches")
                
        # Run artifact injection with patch annotations only
        generated_image = flux_generator.inject_artifacts(
            source_prompt='',
            target_prompt='',
            artifact_type=artifact_type,
            source_img=img_array.copy(),
            output_dir=output_dir,
            reference_patch_indices=reference_patch_indices,
            target_patch_indices=target_patch_indices,
        )
        
        # Create visualizations
        create_flux_visualizations(
            img_array, generated_image, annotation, artifact_type,
            patch_data, img_filename, caption, output_dir, 
            visualizer
        )
        results['artifacts'][artifact_type] = {
            'success': True,
            'class_name': class_name,
            'annotation': annotation,
            'used_patch_annotations': {
                'reference_patches': len(patch_data.get('reference_patch_indices') or []),
                'target_patches': len(patch_data.get('target_patch_indices') or [])

            }
        }
        successful_artifacts += 1
        logger.info(f"  ✅ {artifact_type} artifact generated successfully")
    
    if successful_artifacts > 0:
        results['success'] = True
    
    results['processing_time'] = time.time() - start_time
    return results


def run_flux_generation(segmentation_output_dir: str, artifact_types: List[str],
                       resume: bool = False, device: str = 'cuda',
                       output_dir: Optional[str] = None, inject_step: int=25,
                       pe_step_addition: float=0.3, pe_step_removal: float=0.3, pe_step_distortion: float=0.5,
                       guidance: float=5.0, num_steps: int=25, seed: int=42):
    """Run FLUX artifact generation on processed data"""
    
    # Extract supercategory from output directory name
    supercategory = os.path.basename(segmentation_output_dir).replace('vlpart_output_', '')
    
    # Setup configurations  
    output_dir = output_dir or f'flux_output_{supercategory}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, supercategory)
    logger.info(f"Starting FLUX generation for supercategory: {supercategory}")
    logger.info(f"Reading from: {segmentation_output_dir}")
    logger.info(f"Artifact types: {artifact_types}")
    
    # Setup progress tracking
    progress_file = os.path.join(output_dir, f'flux_progress.json')
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'successful_images': 0,
        'failed_images': 0,
        'artifact_stats': {artifact_type: {'success': 0, 'failure': 0} 
                         for artifact_type in artifact_types},
        'start_time': datetime.now().isoformat(),
        'processed_image_ids': []
    }
    
    # Load previous progress if resuming
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                stats.update(json.load(f))
                logger.info(f"Resuming: {len(stats['processed_image_ids'])} images already processed")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    
    # Setup FLUX configuration
    flux_config = FluxConfig(
        name='flux-dev',
        guidance=guidance,
        num_steps=num_steps,
        pe_step={
            'addition': pe_step_addition,     
            'removal': pe_step_removal,      
            'distortion': pe_step_distortion
        },
        inject_step=inject_step,
        attn_mask_step=0,
        seed=seed,
    )
    
    # Initialize components directly
    logger.info("Initializing FLUX components...")
    flux_generator = FluxGenerator(device=device, config=flux_config)
    visualizer = ImageVisualizer()
    
    try:
        # Get list of processed data files
        processed_data_dir = os.path.join(segmentation_output_dir, 'processed_data')
        if not os.path.exists(processed_data_dir):
            logger.error(f"Processed data directory does not exist: {processed_data_dir}")
            return
            
        data_files = glob.glob(os.path.join(processed_data_dir, 'image_*.pkl'))
        stats['total_images'] = len(data_files)
        logger.info(f"Found {len(data_files)} processed data files")
        
        # Filter out already processed images if resuming
        if resume:
            processed_ids = set(stats['processed_image_ids'])
            data_files = [f for f in data_files 
                         if int(os.path.basename(f).replace('image_', '').replace('.pkl', '')) not in processed_ids]
            logger.info(f"Remaining to process: {len(data_files)} files")
        
        if not data_files:
            logger.info("No files to process!")
            return
            
        # Process files with progress bar
        with tqdm(total=len(data_files), desc=f"FLUX generation for {supercategory} images") as pbar:
            for data_file in data_files:
                result = process_single_image(
                    data_file, flux_generator, visualizer, artifact_types, output_dir, logger
                )
                
                # Update stats
                stats['processed_images'] += 1
                stats['processed_image_ids'].append(result['image_id'])
                
                if result['success']:
                    stats['successful_images'] += 1
                else:
                    stats['failed_images'] += 1
                
                # Update artifact stats
                for artifact_type, artifact_result in result['artifacts'].items():
                    if artifact_result.get('success', False):
                        stats['artifact_stats'][artifact_type]['success'] += 1
                    else:
                        stats['artifact_stats'][artifact_type]['failure'] += 1
                
                # Update progress bar
                status = "✅" if result['success'] else "❌"
                pbar.set_postfix({
                    'Current': f"{status} {result['filename'][:20]}...",
                    'Success': stats['successful_images'],
                    'Failed': stats['failed_images']
                })
                pbar.update(1)
                
                # Save progress periodically
                if stats['processed_images'] % 10 == 0:
                    with open(progress_file, 'w') as f:
                        json.dump(stats, f, indent=2)
                        
        # Final progress save and summary
        with open(progress_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FLUX ARTIFACT GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Supercategory: {supercategory}")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Processed: {stats['processed_images']}")
        logger.info(f"Successful: {stats['successful_images']}")
        logger.info(f"Failed: {stats['failed_images']}")
        success_rate = stats['successful_images']/max(stats['processed_images'], 1)*100
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("\nArtifact Generation Statistics:")
        for artifact_type, artifact_stats in stats['artifact_stats'].items():
            total = artifact_stats['success'] + artifact_stats['failure']
            success_rate = artifact_stats['success'] / max(total, 1) * 100
            logger.info(f"  {artifact_type.title()}: {artifact_stats['success']}/{total} ({success_rate:.1f}%)")
            
        elapsed_time = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()
        logger.info(f"\nTotal time: {elapsed_time/3600:.1f} hours")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("="*60)
        
    finally:
        # Cleanup components
        flux_generator.unload_models()


def main():
    """Main function for FLUX artifact generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate FLUX artifacts from VLPart processing results')
    parser.add_argument('segmentation_output_dir', type=str, 
                       help='Directory containing VLPart processing results')
    parser.add_argument('--artifact-types', nargs='+', 
                       default=['distortion', 'removal', 'addition'],
                       help='Artifact types to generate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: flux_output_{supercategory})')
    parser.add_argument('--inject', type=int, default=25,
                       help='Inject step for FLUX generation (default: 25)')
    parser.add_argument('--pe-step-addition', type=float, default=0.3,
                       help='PE step for addition artifacts (default: 0.3)')
    parser.add_argument('--pe-step-removal', type=float, default=0.3,
                       help='PE step for removal artifacts (default: 0.3)')
    parser.add_argument('--pe-step-distortion', type=float, default=0.5,
                       help='PE step for distortion artifacts (default: 0.5)')
    parser.add_argument('--guidance', type=float, default=5.0,
                       help='Guidance for FLUX generation (default: 5.0)')
    parser.add_argument('--num-steps', type=int, default=25,
                       help='Number of steps for FLUX generation (default: 25)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for FLUX generation (default: 42)')
    args = parser.parse_args()
    
    # Validate VLPart output directory
    if not os.path.exists(args.segmentation_output_dir):
        print(f"❌ Error: VLPart output directory does not exist: {args.segmentation_output_dir}")
        sys.exit(1)
        
    processed_data_dir = os.path.join(args.segmentation_output_dir, 'processed_data')
    if not os.path.exists(processed_data_dir):
        print(f"❌ Error: Processed data directory does not exist: {processed_data_dir}")
        print("   Please run VLPart processing first.")
        sys.exit(1)
    
    run_flux_generation(
        segmentation_output_dir=args.segmentation_output_dir,
        artifact_types=args.artifact_types,
        resume=args.resume,
        device=args.device,
        output_dir=args.output_dir,
        inject_step=args.inject,
        pe_step_addition=args.pe_step_addition,
        pe_step_removal=args.pe_step_removal,
        pe_step_distortion=args.pe_step_distortion,
        guidance=args.guidance,
        num_steps=args.num_steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 