#!/usr/bin/env python3
"""
VLPart Segmentation Batch Processing Script

This script processes COCO images to generate VLPart segmentation results
and saves unified data for later artifact generation.

Features:
- Process all images in a supercategory
- Generate VLPart segmentation results
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

from pipeline import VLPartDetector, COCODataLoader, ImageNetDataLoader, InstanceProcessor, ImageVisualizer
from flux.artifacts_util import mask_to_patch_indices
from pipeline.prompts import artifact_type_decision, addition_sugget_direction

def setup_logging(output_dir: str, supercategory: str):
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'vlpart_processing_{supercategory}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def save_unified_data(img_id: int, data: Dict, output_dir: str):
    """Save all data in a single unified format"""
    output_file = os.path.join(output_dir, 'processed_data', f'image_{img_id}.pkl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert numpy arrays in annotations to lists for better compatibility
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

    # Prepare unified data structure
    unified_data = {
        'image_info': data['image_info'],
        'image_array': data['image_array'],
        'vocab': data['vocab'],
        'predictions': data['predictions'],
        'caption': data['caption'],
        'visualized_output': data['visualized_output'],
        'artifacts': {}  # Unified artifact data
    }
    
    # Combine annotations, masks, and patch data per artifact type
    for artifact_type in data.get('annotations', {}):
        if 'error' not in data['annotations'][artifact_type]:
            unified_data['artifacts'][artifact_type] = {
                'annotation': convert_numpy(data['annotations'][artifact_type]['annotation']),
                'class_name': data['annotations'][artifact_type]['class_name'],
                'masks': data['masks_data'][artifact_type],
                'patch_data': convert_numpy(data['patch_annotations'][artifact_type]),
                'sampled_instance_info': convert_numpy(data['annotations'][artifact_type]['sampled_instance_info'])
            }
        else:
            unified_data['artifacts'][artifact_type] = {
                'error': data['annotations'][artifact_type]['error']
            }
    
    unified_data['processing_timestamp'] = datetime.now().isoformat()
    
    # Save as pickle for efficiency
    with open(output_file, 'wb') as f:
        pickle.dump(unified_data, f)


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
                        output_dir: str, visualizer):
    """Create all visualizations for an image"""
    # Original image
    visualizer.show_image(
        img_array, caption, title="Original Image", 
        image_name=img_filename, 
        base_dir=output_dir,
        filename="01_original_image.png"
    )
    
    # Detection results
    visualizer.show_detection_results(
        img_array, visualized_output,
        image_name=img_filename, 
        base_dir=output_dir,
        filename="02_detection_results.png"
    )
    
    # Artifact-specific visualizations
    for artifact_type, artifact_data in artifacts.items():
        if 'error' not in artifact_data:

            masks = artifact_data['masks'].copy()  # Make a copy to avoid modifying original
            patch_data = artifact_data.get('patch_data', {})
            
            # For addition artifacts, generate target mask from patch indices
            if artifact_type == 'addition' and 'target_patch_indices' in patch_data:
                target_patch_indices = patch_data['target_patch_indices']
                if target_patch_indices:
                    target_mask = create_target_mask_from_patches(
                        target_patch_indices, img_array.shape, patch_size=16
                    )
                    masks['target_mask'] = target_mask
                    print(f"Generated target mask from {len(target_patch_indices)} patch indices for {artifact_type}")
            
            # Create patch mask visualizations
            InstanceProcessor.visualize_patch_masks(
                img_array, {artifact_type: masks}, 
                img_filename, output_dir
            )


def process_single_image(img_info: Dict, vlpart_detector: VLPartDetector, 
                        data_loader, visualizer: ImageVisualizer,
                        artifact_types: List[str], output_dir: str, 
                        config: Dict, openai_client: openai.OpenAI,
                        logger) -> Dict:
    """Process a single image with VLPart segmentation"""
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

        # Generate vocabulary and detect parts
        logger.info(f"  Generating vocabulary and detecting parts...")
        
        # Get categories based on dataset type
        if hasattr(data_loader, 'get_image_categories'):
            # COCO dataset
            categories = data_loader.get_image_categories(img_info)
        else:
            # ImageNet dataset
            class_name = img_info.get('class_name', 'object')
            categories = [class_name]
        
        
        # Generate subpart vocabulary
        vocab = vlpart_detector.generate_subpart_vocab(img_array)
        # Setup VLPart model with vocabulary and detect parts
        vlpart_detector.setup_model(vocab)
        predictions, visualized_output = vlpart_detector.detect_parts(img_array)
        
        # Process each artifact type
        annotations = {}
        masks_data = {}
        patch_annotations = {}
        artifact_count = 0
        
        logger.info(f"  Creating artifact data...")
        
        # Sample target part
        sampled_instance, sampled_idx, class_name = vlpart_detector.sample_target_part(
            predictions, vocab, config['min_area_ratio'], config['max_area_ratio']
        )
        
        if sampled_instance is None:
            results['error'] = "No valid target parts found after filtering"
            return results
        
        # artifact_decision = artifact_type_decision(openai_client, sampled_instance, img_array)
        # artifact_type = artifact_decision['artifact_type']
        artifact_type = random.choice(['addition', 'removal', 'distortion'])
        if artifact_type == 'addition':
            artifact_direction = addition_sugget_direction(openai_client, sampled_instance, class_name, img_array)['direction']
        else:
            artifact_direction = None
        
        # Create masks and patch annotations in one operation
        masks, patch_annot = InstanceProcessor.create_masks_and_patch_annotations_from_instance(
            sampled_instance, img_array.shape, artifact_type
        )
        
        # Create annotation with smart bbox generation (and target mask + patch mapping for addition)
        annotation, patch_mapping = InstanceProcessor.create_annotation_dict(
            instance=sampled_instance,
            img_shape=img_array.shape,
            vocab=vocab,
            part_class_idx=sampled_instance.pred_classes.item(),
            prompt=caption,
            artifact_type=artifact_type,
            artifact_direction=artifact_direction,
            predictions=predictions,  # Pass predictions for smart bbox generation
            max_ref_overlap=config['max_ref_overlap'],
            min_entity_overlap=config['min_entity_overlap']
        )
            
        # Handle patch mapping data
        if patch_mapping is not None:
            # Use the patch mapping results for addition artifacts
            patch_annot['target_patch_indices'] = patch_mapping['target_patch_indices']
            patch_annot['reference_patch_indices'] = patch_mapping['reference_patch_indices']
        
        # Store results
        annotations[artifact_type] = {
            'annotation': annotation,
            'class_name': class_name,
            'sampled_instance_info': {
                'sampled_idx': sampled_idx,
                'bbox_coords': sampled_instance.pred_boxes.tensor[0].cpu().numpy().tolist(),
                'score': sampled_instance.scores.item(),
                'class_idx': sampled_instance.pred_classes.item()
            }
        }
        masks_data[artifact_type] = masks
        patch_annotations[artifact_type] = patch_annot
        artifact_count += 1
        
        logger.info(f"  ✅ {artifact_type} artifact created")
        
        # Prepare unified data structure
        unified_data = {
            'image_info': img_info,
            'image_array': img_array,
            'vocab': vocab,
            'predictions': predictions,
            'caption': caption,
            'visualized_output': visualized_output,
            'annotations': annotations,
            'masks_data': masks_data,
            'patch_annotations': patch_annotations
        }
        
        # Save unified data
        save_unified_data(img_id, unified_data, output_dir)
        
        # Create visualizations
        artifacts_for_viz = {}

        artifacts_for_viz[artifact_type] = {
            'annotation': annotations[artifact_type]['annotation'],
            'masks': masks_data[artifact_type],
            'patch_data': patch_annotations[artifact_type]
        }

        create_visualizations(
            img_array, img_filename, '', visualized_output,
            artifacts_for_viz, output_dir, visualizer
        )
        
        results['success'] = True
        results['artifacts_created'] = artifact_count
        logger.info(f"✅ Processed image {img_id} with {artifact_count} artifacts")
        
    except Exception as e:
        logger.error(f"❌ Error processing image {img_id}: {str(e)}")
        logger.error(traceback.format_exc())
        results['error'] = str(e)
    
    results['processing_time'] = time.time() - start_time
    return results


def run_vlpart_processing(categories: List[str], artifact_types: List[str],
                        max_images: Optional[int] = None, resume: bool = False,
                        config: Dict = None,
                        openai_client: Optional[openai.OpenAI] = None):
    """Run VLPart segmentation processing on all images"""
    
    # Determine dataset type based on config
    dataset_type = config.get('dataset_type', 'coco')
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    category_str = "-".join(categories)
    logger = setup_logging(output_dir, f"{dataset_type}_{category_str}")
    logger.info(f"Starting VLPart processing for {dataset_type} dataset")
    logger.info(f"Categories: {categories}")
    logger.info(f"Artifact types: {artifact_types}")
    logger.info(f"Using smart IoU-based bbox generation (max_ref_overlap={config['max_ref_overlap']}, min_entity_overlap={config['min_entity_overlap']})")
    
    # Setup progress tracking
    progress_file = os.path.join(output_dir, f'processing_progress.json')
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'successful_images': 0,
        'failed_images': 0,
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
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Initialize data loader based on dataset type
    if dataset_type == "coco":
        data_loader = COCODataLoader(config['dataset_path'], config['image_path'])
    elif dataset_type == "imagenet":
        data_loader = ImageNetDataLoader(config['dataset_path'], config['imagenet_split'])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Initialize VLPart detector
    vlpart_detector = VLPartDetector(
        config_file=config.get('vlpart_config_file'),
        model_weights=config.get('vlpart_model_weights'),
        confidence_threshold=config.get('vlpart_confidence_threshold', 0.6),
        openai_client=openai_client or openai.OpenAI()
    )
    
    # Initialize visualizer
    visualizer = ImageVisualizer()
    
    try:
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
        
        # Shuffle and limit
        import random
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
                    img_info, vlpart_detector, data_loader, visualizer,
                    artifact_types, output_dir, config, openai_client, logger
                )
                
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
        logger.info("VLPART PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Dataset: {dataset_type}")
        logger.info(f"Categories: {categories}")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Processed: {stats['processed_images']}")
        logger.info(f"Successful: {stats['successful_images']}")
        logger.info(f"Failed: {stats['failed_images']}")
        success_rate = stats['successful_images']/max(stats['processed_images'], 1)*100
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        elapsed_time = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()
        logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("="*60)
        
    finally:
        # Cleanup components
        vlpart_detector.cleanup()


def main():
    """Main function for VLPart segmentation processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate VLPart segmentation results for COCO or ImageNet datasets')
    parser.add_argument('categories', nargs='+', 
                       help='Categories to process. For COCO: supercategories (person, animal, vehicle, etc.). For ImageNet: class names (dog, cat, car, etc.)')
    parser.add_argument('--dataset', type=str, choices=['coco', 'imagenet'], default='coco',
                       help='Dataset type to use (default: coco)')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to dataset. For COCO: annotations directory. For ImageNet: root directory containing train/val folders')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Path to images (COCO only). For ImageNet, use --imagenet-split instead')
    parser.add_argument('--imagenet-split', type=str, choices=['train', 'val'], default='train',
                       help='ImageNet split to use (train or val)')
    parser.add_argument('--artifact-types', nargs='+', 
                       default=['distortion', 'removal', 'addition'],
                       help='Artifact types to prepare annotations for')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--min-area-ratio', type=float, default=0.05,
                       help='Minimum area ratio for part filtering (default: 0.05)')
    parser.add_argument('--max-area-ratio', type=float, default=0.8,
                       help='Maximum area ratio for part filtering (default: 0.8)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated based on dataset and categories)')
    parser.add_argument('--max-ref-overlap', type=float, default=0.3,
                       help='Maximum allowed overlap with reference bbox for smart addition (default: 0.3)')
    parser.add_argument('--min-entity-overlap', type=float, default=0.1,
                       help='Minimum required overlap with entity bbox for smart addition (default: 0.1)')
    parser.add_argument('--predefined-vocab', nargs='+', default=None,
                       help='Pre-defined vocabulary list (e.g., --predefined-vocab "person head" "person arm" "person leg"). If provided, skips OpenAI API calls for vocabulary generation.')
    
    args = parser.parse_args()

    if args.predefined_vocab:
        args.predefined_vocab = [vocab.replace('_', ' ') for vocab in args.predefined_vocab]
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Set default paths based on dataset type
    if args.dataset == 'coco':
        dataset_path = args.dataset_path or "../../data/coco_2017_extracted/annotations/"
        image_path = args.image_path or "../../data/coco_2017_extracted/train2017/"
        output_dir = args.output_dir or f'vlpart_output_coco_{"-".join(args.categories)}'
        
        # Setup config for COCO
        config = {
            'dataset_type': 'coco',
            'dataset_path': dataset_path,
            'image_path': image_path,
            'super_categories': args.categories,
            'output_dir': output_dir,
            'device': args.device,
            'min_area_ratio': args.min_area_ratio,
            'max_area_ratio': args.max_area_ratio,
            'max_ref_overlap': args.max_ref_overlap,
            'min_entity_overlap': args.min_entity_overlap,
            'predefined_vocab': args.predefined_vocab,
            'vlpart_config_file': None,
            'vlpart_model_weights': None,
            'vlpart_confidence_threshold': 0.6
        }
        
    elif args.dataset == 'imagenet':
        dataset_path = args.dataset_path or "../../data/imagenet/"
        output_dir = args.output_dir or f'vlpart_output_imagenet_{"-".join(args.categories)}'
        
        # Setup config for ImageNet
        config = {
            'dataset_type': 'imagenet',
            'dataset_path': dataset_path,
            'imagenet_split': args.imagenet_split,
            'class_names': args.categories,
            'output_dir': output_dir,
            'device': args.device,
            'min_area_ratio': args.min_area_ratio,
            'max_area_ratio': args.max_area_ratio,
            'max_ref_overlap': args.max_ref_overlap,
            'min_entity_overlap': args.min_entity_overlap,
            'predefined_vocab': args.predefined_vocab,
            'vlpart_config_file': None,
            'vlpart_model_weights': None,
            'vlpart_confidence_threshold': 0.6
        }
    
    print(f"🚀 Starting VLPart processing for {args.dataset.upper()} dataset")
    print(f"📁 Dataset path: {config['dataset_path']}")
    if args.dataset == 'coco':
        print(f"🖼️  Image path: {config['image_path']}")
        print(f"🏷️  Supercategories: {args.categories}")
    else:
        print(f"📊 Split: {args.imagenet_split}")
        print(f"🏷️  Class names: {args.categories}")
    print(f"📤 Output directory: {config['output_dir']}")
    
    try:
        run_vlpart_processing(
            categories=args.categories,
            artifact_types=args.artifact_types,
            max_images=args.max_images,
            resume=args.resume,
            config=config,
            openai_client=openai.OpenAI()
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user.")
        print("Progress saved. Use --resume to continue later.")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main() 