#!/usr/bin/env python3
"""
FLUX Image Reconstruction Batch Processing Script

This script reads unified VLPart processing results and reconstructs
images using FLUX model WITHOUT injecting artifacts. Useful for establishing
baselines and testing FLUX's image reconstruction capabilities.

Features:
- Read unified VLPart processing data
- Reconstruct images using FLUX without artifact injection
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
from tqdm import tqdm
from PIL import Image

from pipeline import FluxGenerator, FluxConfig, ImageVisualizer


def setup_logging(output_dir: str, supercategory: str):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'flux_reconstruction_{supercategory}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def generate_caption_for_image(image_path: str) -> str:
    """Generate a simple caption for the image"""
    # Simple default caption - can be enhanced with vision models if needed
    return "A photorealistic image"


def process_single_image(image_file: str, flux_generator: FluxGenerator,
                        visualizer: ImageVisualizer, 
                        output_dir: str, logger) -> Dict:
    """Process a single image with FLUX reconstruction (no artifact injection)"""
    # Extract image ID from directory name
    image_dir = os.path.dirname(image_file)
    img_id = os.path.basename(image_dir)
    img_filename = os.path.basename(image_file)
    
    logger.info(f"Processing FLUX reconstruction for image {img_id}: {img_filename}")
    
    results = {
        'image_id': img_id,
        'filename': img_filename,
        'success': False,
        'error': None,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    # Load image directly
    try:
        img_array = Image.open(image_file)
    except Exception as e:
        logger.error(f"Failed to load image {image_file}: {e}")
        return {'success': False, 'error': f'Failed to load image: {e}', 
                'image_id': img_id, 'filename': img_filename, 'processing_time': 0}
    
    # Generate caption for the image
    caption = generate_caption_for_image(image_file)
    
    # Create image-specific output directory for FLUX results
    flux_output_path = os.path.join(output_dir, img_id)
    os.makedirs(flux_output_path, exist_ok=True)
    
    # Copy original image to output directory
    flux_viz_path = os.path.join(flux_output_path, "real_image.png")
    if not os.path.exists(flux_viz_path):
        import shutil
        shutil.copy2(image_file, flux_viz_path)
    
    # Reconstruct image without artifacts
    # try:
    logger.info(f"  Reconstructing image without artifacts...")
    
    # Reconstruct image using FLUX with same prompt (no artifact injection)
    # We pass None for artifact_data to skip artifact injection
    reconstructed_image = flux_generator.inject_artifacts(
        source_prompt=caption,
        target_prompt=caption,
        artifact_data=None,  # No artifacts - just reconstruct
        source_img=img_array.copy(),
        use_fireflow=False
    )
    
    # # Save comparison visualization
    # visualizer.show_comparison(
    #     img_array, reconstructed_image, None, caption,
    #     base_dir=flux_output_path,
    #     filename=f"comparison.png",
    # )
    
    # Save reconstructed image
    reconstructed_image.save(os.path.join(flux_output_path, f'reconstructed_image.png'))
    
    results['success'] = True
    logger.info(f"  ✅ Image reconstructed successfully")
        
    # except Exception as e:
    #     logger.error(f"  ❌ Failed to reconstruct image: {str(e)}")
    #     results['error'] = str(e)
    #     results['success'] = False
    
    results['processing_time'] = time.time() - start_time
    return results


def run_flux_reconstruction(input_dir: str,
                           resume: bool = False, device: str = 'cuda',
                           output_dir: Optional[str] = None, inject_step: int=20,
                           guidance: float=5.0, num_steps: int=25, seed: int=42, 
                           use_rf_solver: bool=False, gpu_id: Optional[int] = None,
                           total_gpus: int = 1):
    """Run FLUX image reconstruction on real_image.png files (without artifact injection)
    
    Args:
        gpu_id: GPU ID to use (for multi-GPU parallelization)
        total_gpus: Total number of GPUs being used (for splitting work)
    """
    
    # Set GPU device if specified
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda:0'  # Always use device 0 after setting CUDA_VISIBLE_DEVICES
    
    # Extract directory name for naming purposes
    dir_basename = os.path.basename(input_dir.rstrip('/'))
    
    # Setup configurations  
    output_dir = output_dir or f'flux_reconstruction_{dir_basename}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Add GPU suffix to logger for multi-GPU runs
    logger_suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
    
    # Setup logging
    logger = setup_logging(output_dir, dir_basename + logger_suffix)
    logger.info(f"Starting FLUX reconstruction for directory: {dir_basename}")
    logger.info(f"Reading from: {input_dir}")
    logger.info(f"Mode: Image reconstruction WITHOUT artifact injection")
    if gpu_id is not None:
        logger.info(f"GPU ID: {gpu_id} (part of {total_gpus} GPUs)")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Setup progress tracking (GPU-specific for parallel runs)
    progress_suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
    progress_file = os.path.join(output_dir, f'flux_reconstruction_progress{progress_suffix}.json')
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'successful_images': 0,
        'failed_images': 0,
        'start_time': datetime.now().isoformat(),
        'processed_image_ids': [],
        'gpu_id': gpu_id
    }
    
    # Load previous progress if resuming
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                stats.update(json.load(f))
                logger.info(f"Resuming: {len(stats['processed_image_ids'])} images already processed")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    
    # Setup FLUX configuration (keeping same config as artifact generation)
    flux_config = FluxConfig(
        name='flux-dev',
        guidance=guidance,
        num_steps=num_steps,
        pe_step={
            'addition': 0,      # Not used for reconstruction
            'removal': 0,       # Not used for reconstruction
            'distortion': 0,    # Not used for reconstruction
            'fusion': 0         # Not used for reconstruction
        },
        inject_step=inject_step,
        attn_mask_step=0,
        seed=seed,
        use_rf_solver=use_rf_solver,
    )
    
    # Initialize components directly
    logger.info("Initializing FLUX components...")
    flux_generator = FluxGenerator(device=device, config=flux_config)
    
    # Log configuration
    logger.info("FLUX Configuration")
    logger.info(f"Inject step: {inject_step}")
    logger.info(f"Guidance: {guidance}")
    logger.info(f"Number of steps: {num_steps}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Use RF solver: {use_rf_solver}")
    
    visualizer = ImageVisualizer()
    
    try:
        # Get list of real_image.png files from subdirectories
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return
            
        # Look for subdirectories containing real_image.png files
        image_files = []
        for subdir in os.listdir(input_dir):
            subdir_path = os.path.join(input_dir, subdir)
            if os.path.isdir(subdir_path):
                real_image_file = os.path.join(subdir_path, 'real_image.png')
                if os.path.exists(real_image_file):
                    image_files.append(real_image_file)
        
        stats['total_images'] = len(image_files)
        logger.info(f"Found {len(image_files)} real_image.png files")
        
        # Split work among GPUs if doing multi-GPU processing
        if gpu_id is not None and total_gpus > 1:
            # Sort for consistent splitting
            image_files = sorted(image_files)
            # Assign images to this GPU using round-robin
            image_files = [f for i, f in enumerate(image_files) if i % total_gpus == gpu_id]
            logger.info(f"GPU {gpu_id}: Assigned {len(image_files)} images (out of {stats['total_images']} total)")
            stats['total_images'] = len(image_files)
        
        # Filter out already processed images if resuming
        if resume:
            processed_ids = set(stats['processed_image_ids'])
            image_files = [f for f in image_files 
                         if os.path.basename(os.path.dirname(f)) not in processed_ids]
            logger.info(f"Remaining to process: {len(image_files)} files")
        
        if not image_files:
            logger.info("No files to process!")
            return
            
        # Process files with progress bar
        with tqdm(total=len(image_files), desc=f"FLUX reconstruction for {dir_basename} images") as pbar:
            for image_file in image_files:
                result = process_single_image(
                    image_file, flux_generator, visualizer, output_dir, logger
                )
                
                if result is None:
                    continue
                
                # Update stats
                stats['processed_images'] += 1
                stats['processed_image_ids'].append(result['image_id'])
                
                if result['success']:
                    stats['successful_images'] += 1
                else:
                    stats['failed_images'] += 1
                
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
        logger.info("FLUX IMAGE RECONSTRUCTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Directory: {dir_basename}")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Processed: {stats['processed_images']}")
        logger.info(f"Successful: {stats['successful_images']}")
        logger.info(f"Failed: {stats['failed_images']}")
        success_rate = stats['successful_images']/max(stats['processed_images'], 1)*100
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        elapsed_time = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()
        logger.info(f"\nTotal time: {elapsed_time/3600:.1f} hours")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("="*60)
        
    finally:
        # Cleanup components
        flux_generator.unload_models()


def main():
    """Main function for FLUX image reconstruction"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reconstruct images using FLUX without artifact injection (baseline generation)'
    )
    parser.add_argument('input_dir', type=str, 
                       help='Directory containing subdirectories with real_image.png files')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: flux_reconstruction_{dirname})')
    parser.add_argument('--inject', type=int, default=25,
                       help='Inject step for FLUX generation (default: 25)')
    parser.add_argument('--guidance', type=float, default=5.0,
                       help='Guidance for FLUX generation (default: 5.0)')
    parser.add_argument('--num-steps', type=int, default=25,
                       help='Number of steps for FLUX generation (default: 25)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for FLUX generation (default: 42)')
    parser.add_argument('--use-rf-solver', action='store_true',
                       help='Use RF solver (second-order) instead of first-order denoising (default: False)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU ID to use (for multi-GPU parallelization)')
    parser.add_argument('--total-gpus', type=int, default=1,
                       help='Total number of GPUs being used (for splitting work)')
    
    args = parser.parse_args()
    
    run_flux_reconstruction(
        input_dir=args.input_dir,
        resume=args.resume,
        device=args.device,
        output_dir=args.output_dir,
        inject_step=args.inject,
        guidance=args.guidance,
        num_steps=args.num_steps,
        seed=args.seed,
        use_rf_solver=args.use_rf_solver,
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus
    )


if __name__ == "__main__":
    main()

