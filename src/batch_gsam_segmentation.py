"""
GSAM (Grounded SAM) Segmentation Batch Processing Script

This script processes COCO images to generate GSAM segmentation results
and saves unified data for later artifact generation.

Features:
- Process all images in a supercategory
- Generate GSAM segmentation results  
- Create target part annotations for each artifact type
- Generate and save reference/target masks with patch granularity
- Save unified results for later processing
- Progress tracking and resumption capability
"""

import os
import sys
import json
import time
import logging
import pickle
import random

import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import traceback

import numpy as np
import cv2
import torch
import openai
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from pipeline import GSAMDetector, COCODataLoader, ImageNetDataLoader, CustomDirectoryDataLoader, InstanceProcessor, ImageVisualizer
from pipeline.prompts import addition_select_candidate, visualize_all_candidates, visualize_candidate_images_for_api, addition_suggest_offset

def setup_logging(output_dir: str, supercategory: str):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'gsam_processing_{supercategory}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)



def create_target_mask_from_patches(target_patch_indices: List[int], img_shape: Tuple, patch_size: int = 16) -> np.ndarray:
    """
    Create target mask from patch indices for addition artifacts
    
    Args:
        target_patch_indices: List of target patch indices
        img_shape: Image shape (H, W, C)
        patch_size: Size of patches (default 16)
        
    Returns:
        Target mask as numpy array
    """
    from flux.artifacts_util import patch_indices_to_coords
    
    H, W = img_shape[:2]
    patch_W = W // patch_size
    
    # Create empty mask
    target_mask = np.zeros((H, W), dtype=np.uint8)
    
    if target_patch_indices and patch_indices_to_coords is not None:
        # Convert patch indices to coordinates
        patch_coords = patch_indices_to_coords(target_patch_indices, patch_W, txt_len=512)
        
        # Fill patches in the mask
        for py, px in patch_coords:
            y_start = py * patch_size
            y_end = min((py + 1) * patch_size, H)
            x_start = px * patch_size
            x_end = min((px + 1) * patch_size, W)
            
            target_mask[y_start:y_end, x_start:x_end] = 255
    
    return target_mask


def create_visualizations(img_array: np.ndarray, img_filename: str, caption: str, 
                        visualized_output: np.ndarray, artifacts: Dict, 
                        image_output_dir: str, visualizer):
    """Create all visualizations for an image in image-specific directory"""
    # Original image
    visualizer.show_image(
        # img_array, caption, title="Original Image", 
        img_array, caption,
        image_name=img_filename, 
        base_dir=image_output_dir,
        filename="01_original_image.png"
    )
    
    # Detection results
    visualizer.show_detection_results(
        img_array, visualized_output,
        image_name=img_filename, 
        base_dir=image_output_dir,
        filename="02_detection_results.png"
    )
    
    # Artifact-specific visualizations
    for artifact_type, artifact_data in artifacts.items():
        if 'error' not in artifact_data:

            patch_data = artifact_data.get('patch_data', {})
            
            if 'target_patch_indices' in patch_data:
                target_mask = create_target_mask_from_patches(
                    patch_data['target_patch_indices'], img_array.shape, patch_size=16
                )
            if 'reference_patch_indices' in patch_data:
                reference_mask = create_target_mask_from_patches(
                    patch_data['reference_patch_indices'], img_array.shape, patch_size=16
            )
            
            # Create patch mask visualizations
            InstanceProcessor.visualize_patch_masks(
                img_array, {artifact_type: {
                    'reference_mask': reference_mask,
                    'target_mask': target_mask
                }}, 
                img_filename, image_output_dir
            )


def process_single_image(img_info: Dict, gsam_detector: GSAMDetector, 
                        data_loader, visualizer: ImageVisualizer,
                        output_dir: str, 
                        config: Dict, openai_client: openai.OpenAI,
                        logger) -> Dict:
    """Process a single image with GSAM segmentation"""
    img_id = img_info['id']
    img_filename = img_info['file_name']
    
    logger.info(f"Processing image {img_id}: {img_filename}")
    
    results = {
        'image_id': img_id,
        'filename': img_filename,
        'success': False,
        'error': None,
        'processing_time': 0,
        'artifacts_created': 0
    }
    
    start_time = time.time()
    
    try:
        # Load image and generate caption using data loader
        if hasattr(data_loader, 'load_image_by_info'):
            img_array = data_loader.load_image_by_info(img_info)
        else:
            img_array = data_loader.load_image_by_path(img_info['file_path'])
        
        caption = data_loader.get_image_caption(img_info)

        # Try artifact types specified in config
        artifact_types_to_try = config['artifact_types']
        successful_artifact_type = None
        sampled_instance = None
        sampled_idx = None
        class_name = None
        vocab = None
        predictions = None
        visualized_output = None
        
        annotations = {}
        masks_data = {}
        patch_annotations = {}
        artifact_count = 0
        successful_artifact_type = None
        


        for artifact_type in artifact_types_to_try:
            try:
                logger.info(f"  Trying artifact type: {artifact_type}")
                image_output_dir = os.path.join(output_dir, f'image_{img_id}')
                os.makedirs(image_output_dir, exist_ok=True)
                # Step 1: Generate subpart vocabulary from the image
                logger.info(f"    Step 1: Generating subpart vocabulary for {artifact_type}...")
                vocab = gsam_detector.generate_subpart_vocab(img_array, artifact_type)

                # Step 2: Use detect_parts to detect instances according to the vocab
                logger.info(f"    Step 2: Detecting parts using Grounded SAM...")
                predictions, visualized_output = gsam_detector.detect_parts(img_array, vocab)
                
                # Step 3: Sample target part from detection results
                logger.info(f"    Step 3: Sampling target part for {artifact_type}...")
                sampled_instance, sampled_idx, class_name = gsam_detector.sample_target_part(
                    predictions, vocab, config['min_area_ratio'], config['max_area_ratio']
                )

                if sampled_instance is not None:
                    successful_artifact_type = artifact_type
                    logger.info(f"  ✅ Successfully found target part for {artifact_type}")
                    
                    # Immediately create artifact data for the successful type
                    logger.info(f"  Creating artifact data...")
                    
                    # Create masks and patch annotations in one operation
                    patch_annot = InstanceProcessor.create_masks_and_patch_annotations_from_instance(
                        sampled_instance, img_array.shape, artifact_type, patch_size=16
                    )
                    
                    if artifact_type == 'distortion':
                        if config['distortion_kernel'] == 'none':
                            distortion_kernel = random.choice(['none', 'jitter', 'swirl', 'voronoi'])
                        else:
                            distortion_kernel = config['distortion_kernel']

                    logger.info(f"  Randomly selected distortion kernel: {distortion_kernel}")
                    
                    target_patches, reference_patches = InstanceProcessor.create_artifact_patches(
                        artifact_type, 
                        sampled_instance, 
                        predictions, 
                        patch_annot, 
                        openai_client, 
                        vocab, 
                        class_name, 
                        img_array, 
                        16, 
                        distortion_kernel=distortion_kernel,
                        output_dir=image_output_dir,
                        img_filename=img_filename
                    )

                    # Map patches to bounding box coordinates in real image dimensions
                    target_bbox = InstanceProcessor.patch_coords_to_bbox(target_patches, patch_size=16)
                    reference_bbox = InstanceProcessor.patch_coords_to_bbox(reference_patches, patch_size=16)
                    
                    logger.info(f"    Target patches: {len(target_patches)} patches -> BBox: {target_bbox}")
                    logger.info(f"    Reference patches: {len(reference_patches)} patches -> BBox: {reference_bbox}")
                    
                    # Store bounding box information for later use
                    patch_annot['target_bbox'] = target_bbox
                    patch_annot['reference_bbox'] = reference_bbox

                    # Create patch mapping first
                    target_patch_indices, reference_patch_indices = InstanceProcessor.patch_coords_to_indices(
                        artifact_type=artifact_type,
                        target_patches=target_patches,
                        reference_patches=reference_patches,
                        patch_annot=patch_annot,
                        img_shape=img_array.shape,
                        patch_size=16
                    )

                    # Update patch annotations with mapping
                    patch_annot['target_patch_indices'] = target_patch_indices
                    patch_annot['reference_patch_indices'] = reference_patch_indices
                    
                    # Create annotation dictionary
                    annotation, patch_annot = InstanceProcessor.create_annotation_dict(
                        instance=sampled_instance,
                        vocab=vocab,
                        patch_annot=patch_annot,
                    )
                    
                    # Store results
                    annotations[artifact_type] = {
                        'annotation': annotation,
                        'entity_name': vocab[0],
                        'class_name': class_name,
                        'sampled_instance_info': {
                            'sampled_idx': sampled_idx,
                            'bbox_coords': sampled_instance['pred_box'].cpu().numpy().tolist(),
                            'score': sampled_instance['score'].item(),
                            'class_idx': sampled_instance['pred_class'].item()
                        },
                        'patch_data': patch_annot
                    }
                    artifact_count += 1
                    
                    logger.info(f"  ✅ {artifact_type} artifact created")
                    break  # Exit the loop once artifact is successfully created
                else:
                    logger.info(f"No valid target parts found for {artifact_type}")
                    shutil.rmtree(image_output_dir)
                    continue
            except Exception as e:
                logger.info(f"Error with {artifact_type}: {str(e)}")
                shutil.rmtree(image_output_dir)
                continue


        if successful_artifact_type is None:
            results['error'] = "No valid target parts found for any artifact type after filtering"
            return results
                # Create image-specific output directory early so it can be used by create_artifact_patches

        # Prepare unified data structure directly in final format
        unified_data = {
            'image_info': img_info,
            'image_array': img_array,
            'vocab': vocab,
            'caption': caption,
            'artifacts': {}  # Unified artifact data
        }
        
        # Convert numpy arrays for better compatibility
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            else:
                return obj
        
        # Combine annotations, masks, and patch data per artifact type
        for artifact_type in annotations:
            if 'error' not in annotations[artifact_type]:
                unified_data['artifacts'][artifact_type] = {
                    'annotation': convert_numpy(annotations[artifact_type]['annotation']),
                    'class_name': annotations[artifact_type]['class_name'],
                    'masks': masks_data.get(artifact_type, {}),
                    'patch_data': convert_numpy(annotations[artifact_type]['patch_data']),
                    'kernel_type': distortion_kernel,
                    'sampled_instance_info': convert_numpy(annotations[artifact_type]['sampled_instance_info'])
                }
            else:
                unified_data['artifacts'][artifact_type] = {
                    'error': annotations[artifact_type]['error']
                }
        
        unified_data['processing_timestamp'] = datetime.now().isoformat()
        
        # Save unified data directly
        output_file = os.path.join(image_output_dir, 'metadata.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(unified_data, f)
        
        # Create visualizations
        artifacts_for_viz = {}

        artifacts_for_viz[artifact_type] = {
            'annotation': annotations[artifact_type]['annotation'],
            'patch_data': annotations[artifact_type]['patch_data']
        }

        create_visualizations(
            img_array, img_filename, '', visualized_output,
            artifacts_for_viz, image_output_dir, visualizer
        )
        
        results['success'] = True
        results['artifacts_created'] = artifact_count
        results['successful_artifact_type'] = successful_artifact_type
        logger.info(f"✅ Processed image {img_id} with {artifact_count} artifacts")

    except Exception as e:
        logger.error(f"❌ Error processing image {img_id}: {str(e)}")
        logger.error(traceback.format_exc())
        results['error'] = str(e)
    
    results['processing_time'] = time.time() - start_time
    return results


def run_gsam_processing(categories: List[str], artifact_types: List[str],
                        max_images: Optional[int] = None, resume: bool = False,
                        config: Dict = None,
                        openai_client: Optional[openai.OpenAI] = None,
                        seed: Optional[int] = None):
    """Run GSAM segmentation processing on all images"""
    
    # Determine dataset type based on config
    dataset_type = config['dataset_type']
    distortion_kernel = config['distortion_kernel']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    category_str = "-".join(categories)
    logger = setup_logging(output_dir, f"{dataset_type}_{category_str}")
    logger.info(f"Starting GSAM processing for {dataset_type} dataset")
    logger.info(f"Categories: {categories}")
    logger.info(f"Artifact types to process: {config['artifact_types']}")
    logger.info(f"Distortion kernel: {distortion_kernel}")
    logger.info(f"Using smart IoU-based bbox generation (max_ref_overlap={config['max_ref_overlap']}, min_entity_overlap={config['min_entity_overlap']})")
    
    # Setup progress tracking
    progress_file = os.path.join(output_dir, f'processing_progress.json')
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'successful_images': 0,
        'failed_images': 0,
        'start_time': datetime.now().isoformat(),
        'processed_image_ids': [],
        'artifact_counts': {
            'addition': 0,
            'removal': 0,
            'distortion': 0
        }
    }
    
    # Load previous progress if resuming
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                stats.update(json.load(f))
                logger.info(f"Resuming: {len(stats['processed_image_ids'])} images already processed")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Initialize data loader based on dataset type
    if dataset_type == "coco":
        data_loader = COCODataLoader(config['dataset_path'], config['image_path'])
    elif dataset_type == "imagenet":
        data_loader = ImageNetDataLoader(config['dataset_path'], config['imagenet_split'])
    elif dataset_type == "custom":
        data_loader = CustomDirectoryDataLoader(config['dataset_path'])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Initialize GSAM detector
    gsam_detector = GSAMDetector(
        grounding_config_file=config['grounding_config_file'],
        grounding_checkpoint=config['grounding_checkpoint'],
        sam_version=config['sam_version'],
        sam_checkpoint=config['sam_checkpoint'],
        sam_hq_checkpoint=config['sam_hq_checkpoint'],
        use_sam_hq=config['use_sam_hq'],
        box_threshold=config['box_threshold'],
        text_threshold=config['text_threshold'],
        bert_base_uncased_path=config['bert_base_uncased_path'],
        device=config['device'],
        openai_client=openai_client or openai.OpenAI()
    )
    
    # Initialize visualizer
    visualizer = ImageVisualizer()
    
    # Get image list
    logger.info("Loading image list...")
    if dataset_type == "coco":
        cat_ids = data_loader.get_category_ids(categories)
        image_list = []
        image_ids_seen = set()
        
        # Count instances per image if filtering is requested
        max_instances_per_image = 3
        instance_counts = {}
        if max_instances_per_image is not None:
            print("Counting instances per image...")
            from collections import defaultdict
            instance_counts = defaultdict(int)
            for ann in data_loader.coco_class.dataset['annotations']:
                image_id = ann['image_id']
                instance_counts[image_id] += 1
        
        for cat_id in cat_ids:
            img_ids = data_loader.coco_class.getImgIds(catIds=[cat_id])
            for img_id in img_ids:
                if img_id not in image_ids_seen:
                    # Filter by instance count if specified
                    if max_instances_per_image is not None:
                        if instance_counts[img_id] >= max_instances_per_image:
                            continue
                    
                    img_info = data_loader.coco_class.loadImgs([img_id])[0]
                    image_list.append(img_info)
                    image_ids_seen.add(img_id)
                    
    elif dataset_type == "imagenet":
        # Determine target synsets
        target_synsets = []
        for class_name in categories:
            for synset, mapped_name in data_loader.class_mapping.items():
                if mapped_name.lower() == class_name.lower() and synset in data_loader.synsets:
                    target_synsets.append(synset)
        else:
            target_synsets = data_loader.synsets
        
        image_list = []
        for synset in target_synsets:
            image_paths = data_loader.get_images_by_synset(synset)
            for img_path in image_paths:
                img_info = {
                    'id': hash(img_path) % 1000000,  # Generate unique ID
                    'file_name': os.path.basename(img_path),
                    'file_path': img_path,
                    'synset': synset,
                    'class_name': data_loader.class_mapping.get(synset, synset)
                }
                image_list.append(img_info)
                
                if max_images and len(image_list) >= max_images:  
                    break
            if max_images and len(image_list) >= max_images:
                break
                
    elif dataset_type == "custom":
        # Get all available class names from the directory structure
        available_classes = data_loader.get_class_names()
        logger.info(f"Available classes in custom dataset: {available_classes}")
        
        # Filter categories to only include those available in the dataset
        target_classes = [cls for cls in categories if cls in available_classes]
        if not target_classes:
            logger.warning(f"None of the specified categories {categories} found in dataset. Available: {available_classes}")
            target_classes = available_classes  # Use all available classes
        
        logger.info(f"Processing classes: {target_classes}")
        
        image_list = []
        for class_name in target_classes:
            image_paths = data_loader.get_all_images_from_class(class_name)
            for img_path in image_paths:
                img_info = {
                    'id': hash(img_path) % 1000000,  # Generate unique ID
                    'file_name': os.path.basename(img_path),
                    'file_path': img_path,
                    'class_name': class_name
                }
                image_list.append(img_info)
                
                if max_images and len(image_list) >= max_images:
                    break
            if max_images and len(image_list) >= max_images:
                break
    
    # Shuffle and limit
    import random
    if seed is not None:
        random.seed(seed)
    random.shuffle(image_list)
    if max_images:
        image_list = image_list[:max_images]
    
    stats['total_images'] = len(image_list)
    logger.info(f"Processing {len(image_list)} images")
    
    # Filter already processed images
    if resume:
        processed_ids = set(stats['processed_image_ids'])
        image_list = [img for img in image_list if img['id'] not in processed_ids]
        logger.info(f"Remaining to process: {len(image_list)} images")
    
    if not image_list:
        logger.info("No images to process!")
        return
    
    # Process images
    with tqdm(total=len(image_list), desc=f"Processing {dataset_type} images") as pbar:
        for img_info in image_list:
            result = process_single_image(
                img_info, gsam_detector, data_loader, visualizer,
                output_dir, config, openai_client, logger
            )
            
            # Update stats
            stats['processed_images'] += 1
            stats['processed_image_ids'].append(result['image_id'])
            
            if result['success']:
                stats['successful_images'] += 1
                # Track artifact type counts
                if 'successful_artifact_type' in result and result['successful_artifact_type']:
                    artifact_type = result['successful_artifact_type']
                    if artifact_type in stats['artifact_counts']:
                        stats['artifact_counts'][artifact_type] += 1
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
    logger.info("GSAM PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Processed: {stats['processed_images']}")
    logger.info(f"Successful: {stats['successful_images']}")
    logger.info(f"Failed: {stats['failed_images']}")
    success_rate = stats['successful_images']/max(stats['processed_images'], 1)*100
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Print artifact type counts
    logger.info("")
    logger.info("ARTIFACT TYPE BREAKDOWN:")
    logger.info("-" * 40)
    for artifact_type, count in stats['artifact_counts'].items():
        percentage = (count / max(stats['successful_images'], 1)) * 100
        logger.info(f"{artifact_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    elapsed_time = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()
    logger.info("")
    logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("="*60)
    
    # Cleanup components
    gsam_detector.cleanup()


def main():
    """Main function for GSAM segmentation processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate GSAM segmentation results for COCO or ImageNet datasets')
    parser.add_argument('categories', nargs='+', 
                       help='Categories to process. For COCO: supercategories (person, animal, vehicle, etc.). For ImageNet: class names (dog, cat, car, etc.)')
    parser.add_argument('--dataset', type=str, choices=['coco', 'imagenet', 'custom'], default='custom',
                       help='Dataset type to use (default: custom)')
    parser.add_argument('--dataset-path', type=str, default='/home/jhpark/image-artifacts/data/eval_coco_animals',
                       help='Path to dataset. For COCO: annotations directory. For ImageNet: root directory containing train/val folders. For custom: root directory with class subdirectories')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Path to images (COCO only). For ImageNet, use --imagenet-split instead')
    parser.add_argument('--imagenet-split', type=str, choices=['train', 'val'], default='train',
                       help='ImageNet split to use (train or val)')
    parser.add_argument('--artifact-types', nargs='+', 
                       default=['distortion', 'removal', 'addition'],
                       help='Artifact types to process (only specified types will be generated)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--min-area-ratio', type=float, default=0.005,
                       help='Minimum area ratio for part filtering (default: 0.05)')
    parser.add_argument('--max-area-ratio', type=float, default=0.5,
                       help='Maximum area ratio for part filtering (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default='../gsam_output_eval_animals',
                       help='Output directory (default: ../gsam_output_eval_animals)')
    parser.add_argument('--max-ref-overlap', type=float, default=0.3,
                       help='Maximum allowed overlap with reference bbox for smart addition (default: 0.3)')
    parser.add_argument('--min-entity-overlap', type=float, default=0.1,
                       help='Minimum required overlap with entity bbox for smart addition (default: 0.1)')
    parser.add_argument('--predefined-vocab', nargs='+', default=None,
                       help='Pre-defined vocabulary list (e.g., --predefined-vocab "person head" "person arm" "person leg"). If provided, skips OpenAI API calls for vocabulary generation.')
    
    # GSAM-specific arguments
    parser.add_argument('--grounding-config', type=str, default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                       help='Path to GroundingDINO config file')
    parser.add_argument('--grounding-checkpoint', type=str, default='weight/groundingdino_swint_ogc.pth',
                       help='Path to GroundingDINO checkpoint')
    parser.add_argument('--sam-version', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model version (default: vit_h)')
    parser.add_argument('--sam-checkpoint', type=str, default='weight/sam_vit_h_4b8939.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--sam-hq-checkpoint', type=str, default=None,
                       help='Path to SAM-HQ checkpoint')
    parser.add_argument('--use-sam-hq', action='store_true',
                       help='Use SAM-HQ instead of regular SAM')
    parser.add_argument('--box-threshold', type=float, default=0.3,
                       help='Box threshold for GroundingDINO (default: 0.3)')
    parser.add_argument('--text-threshold', type=float, default=0.25,
                       help='Text threshold for GroundingDINO (default: 0.25)')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='NMS threshold for GroundingDINO (default: 0.5)')
    parser.add_argument('--bert-base-uncased-path', type=str, default=None,
                       help='Path to BERT base uncased model')
    parser.add_argument('--distortion-kernel', type=str, default='none', 
                       choices=['none', 'jitter', 'swirl', 'voronoi', 'flip'],
                       help='Type of distortion kernel to apply for distortion artifacts (default: none)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()

    if args.predefined_vocab:
        args.predefined_vocab = [vocab.replace('_', ' ') for vocab in args.predefined_vocab]
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Setup configuration based on dataset type
    if args.dataset == 'coco':
        dataset_path = args.dataset_path or "../../data/coco_2017_extracted/annotations/"
        image_path = args.image_path or "../../data/coco_2017_extracted/train2017/"
        output_dir = args.output_dir or f'gsam_output_coco_{"-".join(args.categories)}'

        # Setup config for COCO
        config = {
            'dataset_type': 'coco',
            'dataset_path': dataset_path,
            'image_path': image_path,
            'super_categories': args.categories,
            'artifact_types': args.artifact_types,
            'output_dir': output_dir,
            'device': args.device,
            'min_area_ratio': args.min_area_ratio,
            'max_area_ratio': args.max_area_ratio,
            'max_ref_overlap': args.max_ref_overlap,
            'min_entity_overlap': args.min_entity_overlap,
            'predefined_vocab': args.predefined_vocab,
            'grounding_config_file': args.grounding_config,
            'grounding_checkpoint': args.grounding_checkpoint,
            'sam_version': args.sam_version,
            'sam_checkpoint': args.sam_checkpoint,
            'sam_hq_checkpoint': args.sam_hq_checkpoint,
            'use_sam_hq': args.use_sam_hq,
            'box_threshold': args.box_threshold,
            'text_threshold': args.text_threshold,
            'bert_base_uncased_path': args.bert_base_uncased_path,
            'distortion_kernel': args.distortion_kernel
        }
        
    elif args.dataset == 'imagenet':
        dataset_path = args.dataset_path or "../../data/imagenet/"
        output_dir = args.output_dir or f'gsam_output_imagenet_{"-".join(args.categories)}'
        
        # Setup config for ImageNet
        config = {
            'dataset_type': 'imagenet',
            'dataset_path': dataset_path,
            'imagenet_split': args.imagenet_split,
            'class_names': args.categories,
            'artifact_types': args.artifact_types,
            'output_dir': output_dir,
            'device': args.device,
            'min_area_ratio': args.min_area_ratio,
            'max_area_ratio': args.max_area_ratio,
            'max_ref_overlap': args.max_ref_overlap,
            'min_entity_overlap': args.min_entity_overlap,
            'predefined_vocab': args.predefined_vocab,
            'grounding_config_file': args.grounding_config,
            'grounding_checkpoint': args.grounding_checkpoint,
            'sam_version': args.sam_version,
            'sam_checkpoint': args.sam_checkpoint,
            'sam_hq_checkpoint': args.sam_hq_checkpoint,
            'use_sam_hq': args.use_sam_hq,
            'box_threshold': args.box_threshold,
            'text_threshold': args.text_threshold,
            'bert_base_uncased_path': args.bert_base_uncased_path,
            'distortion_kernel': args.distortion_kernel
        }
        
    elif args.dataset == 'custom':
        dataset_path = args.dataset_path
        output_dir = args.output_dir
        
        # Setup config for Custom Directory
        config = {
            'dataset_type': 'custom',
            'dataset_path': dataset_path,
            'class_names': args.categories,
            'artifact_types': args.artifact_types,
            'output_dir': output_dir,
            'device': args.device,
            'min_area_ratio': args.min_area_ratio,
            'max_area_ratio': args.max_area_ratio,
            'max_ref_overlap': args.max_ref_overlap,
            'min_entity_overlap': args.min_entity_overlap,
            'predefined_vocab': args.predefined_vocab,
            'grounding_config_file': args.grounding_config,
            'grounding_checkpoint': args.grounding_checkpoint,
            'sam_version': args.sam_version,
            'sam_checkpoint': args.sam_checkpoint,
            'sam_hq_checkpoint': args.sam_hq_checkpoint,
            'use_sam_hq': args.use_sam_hq,
            'box_threshold': args.box_threshold,
            'text_threshold': args.text_threshold,
            'bert_base_uncased_path': args.bert_base_uncased_path,
            'distortion_kernel': args.distortion_kernel
        }
    
    print(f"🚀 Starting GSAM processing for {args.dataset.upper()} dataset")
    print(f"📁 Dataset path: {config['dataset_path']}")
    if args.dataset == 'coco':
        print(f"🖼️  Image path: {config['image_path']}")
        print(f"🏷️  Supercategories: {args.categories}")
    elif args.dataset == 'imagenet':
        print(f"📊 Split: {args.imagenet_split}")
        print(f"🏷️  Class names: {args.categories}")
    elif args.dataset == 'custom':
        print(f"🏷️  Class names: {args.categories}")
    print(f"🎯 Artifact types: {args.artifact_types}")
    print(f"🔧 Distortion kernel: {args.distortion_kernel}")
    print(f"📤 Output directory: {config['output_dir']}")
    
    run_gsam_processing(
        categories=args.categories,
        artifact_types=args.artifact_types,
        max_images=args.max_images,
        resume=args.resume,
        config=config,
        openai_client=openai.OpenAI(),
        seed=args.seed
    )



if __name__ == "__main__":
    def set_seed(seed):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may slow down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.seed is not None:
        set_seed(args.seed)
    main() 