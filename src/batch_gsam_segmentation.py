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
import uuid
import sys
import json
import time
import logging
import pickle
import random
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import traceback
from collections import defaultdict
import random
import numpy as np
import openai
import torch
from tqdm import tqdm

from pipeline import (
    GSAMDetector, InstanceProcessor, ImageVisualizer,
)
from pipeline.data_loader import _initialize_data_loader, _get_image_list
from pipeline.prompts import get_all_entity_subparts


def setup_logging(output_dir: str, supercategory: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        supercategory: Category name for log file naming
        
    Returns:
        Configured logger instance
    """
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_dir, f'gsam_processing_{supercategory}_{timestamp}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def create_visualizations(
    img_array: np.ndarray, 
    img_filename: str, 
    caption: str,
    visualized_output: np.ndarray, 
    artifacts: Dict[str, Any], 
    image_output_dir: str, 
    visualizer: ImageVisualizer
) -> None:
    """
    Create all visualizations for an image in image-specific directory.
    
    Args:
        img_array: Original image array
        img_filename: Image filename
        caption: Image caption (unused but kept for compatibility)
        visualized_output: Detection visualization output
        artifacts: Artifact data dictionary (now supports lists of artifacts per type)
        image_output_dir: Output directory for visualizations
        visualizer: ImageVisualizer instance
    """
    # Save original image
    visualizer.save_raw_image(
        img_array,
        base_dir=image_output_dir,
        filename="real_image.png"
    )
    
    # # Save detection results
    # visualizer.show_detection_results(
    #     img_array, 
    #     visualized_output,
    #     image_name=img_filename, 
    #     base_dir=image_output_dir,
    #     filename="02_detection_results.png"
    # )
    
    # # Create artifact-specific visualizations
    # for artifact_type, artifacts_list in artifacts.items():
    #     if artifacts_list:  # Check if list is not empty
    #         # Handle list of artifacts per type
    #         if not isinstance(artifacts_list, list):
    #             artifacts_list = [artifacts_list]
            
    #         # Create combined visualization for all artifacts of this type
    #         all_target_masks = []
    #         all_reference_masks = []
            
    #         for artifact_idx, artifact_data in enumerate(artifacts_list):
    #             if 'error' not in artifact_data:
    #                 patch_data = artifact_data.get('patch_data', {})
                    
    #                 # Create target mask if patch indices exist
    #                 target_mask = None
    #                 if 'target_patch_indices' in patch_data:
    #                     target_mask = create_target_mask_from_patches(
    #                         patch_data['target_patch_indices'], 
    #                         img_array.shape, 
    #                         patch_size=16
    #                     )
    #                     if target_mask is not None:
    #                         all_target_masks.append((target_mask, artifact_idx))
                    
    #                 # Create reference mask if patch indices exist
    #                 reference_mask = None
    #                 if 'reference_patch_indices' in patch_data:
    #                     reference_mask = create_target_mask_from_patches(
    #                         patch_data['reference_patch_indices'], 
    #                         img_array.shape, 
    #                         patch_size=16
    #                     )
    #                     if reference_mask is not None:
    #                         all_reference_masks.append((reference_mask, artifact_idx))
            
    #         # Create visualizations for multiple artifacts
    #         if all_target_masks and all_reference_masks:
    #             # Create a combined mask structure for visualization
    #             combined_masks = {}
                
    #             for idx, (target_mask, artifact_idx) in enumerate(all_target_masks):
    #                 if idx < len(all_reference_masks):
    #                     reference_mask, _ = all_reference_masks[idx]
                        
    #                     # Create artifact-specific key
    #                     artifact_key = f"{artifact_type}_artifact_{artifact_idx}"
    #                     combined_masks[artifact_key] = {
    #                         'reference_mask': reference_mask,
    #                         'target_mask': target_mask
    #                     }
                
    #             # Visualize all artifacts for this type
    #             if combined_masks:
    #                 InstanceProcessor.visualize_patch_masks(
    #                     img_array, 
    #                     combined_masks, 
    #                     img_filename, 
    #                     image_output_dir
    #                 )


def _try_process_all_artifact_types(
    img_array: np.ndarray,
    img_id: int,
    gsam_detector: GSAMDetector,
    config: Dict[str, Any],
    openai_client: openai.OpenAI,
    image_output_dir: str,
    img_filename: str,
    logger: logging.Logger
) -> Tuple[Dict[str, List[Dict]], Optional[np.ndarray], Optional[Any]]:
    """
    Process all artifact types for an image in a single detection call.
    
    Args:
        img_array: Image array
        img_id: Image ID
        gsam_detector: GSAM detector instance
        config: Configuration dictionary
        openai_client: OpenAI client
        image_output_dir: Output directory
        img_filename: Image filename
        logger: Logger instance
        
    Returns:
        Tuple of (artifacts_by_type, visualized_output, combined_vocab)
    """
    logger.info(f"Getting entity subparts for all artifact types...")

    
    # Step 1: Get all entity subparts from API
    all_subparts_response = get_all_entity_subparts(openai_client, img_array)
    if not all_subparts_response or 'error' in all_subparts_response:
        raise RuntimeError("Failed to get entity subparts")
        
    # Transform the response into a nested dictionary structure
    # First level: entity, Second level: subpart, Value: list of artifact types
    entities = set()
    subentities = set()
    entity_subpart_artifacts = defaultdict(lambda: defaultdict(list))
    for artifact_type, vocab_data in all_subparts_response.items():
        entity = vocab_data['entity']
        subparts = vocab_data['subparts']
        entities.add(entity)
        for subpart in subparts:
            subentities.add(subpart)
            entity_subpart_artifacts[entity][subpart].append(artifact_type)

    # Step 3: Single detection call with combined vocabulary
    logger.info(f"Detecting parts using Grounded SAM...")
    predictions, entity_predictions, visualized_output = gsam_detector.detect_parts(img_array, list(entities), list(subentities), min_area_ratio=0.005, max_area_ratio=0.5)
    # Step 4: Sample target parts with entity-aware logic
    logger.info(f"Sampling target parts across all entities...")

    artifacts = []
    for prediction in predictions:
        entity_name = prediction['mapped_entity_name']
        subentity_name = prediction['subentity_name']
        
        artifact_type = random.choice(entity_subpart_artifacts[entity_name][subentity_name])
        logger.info(f"Creating {artifact_type} artifact for {entity_name} of {subentity_name}")
        try:
            annotation = _create_artifact_annotations(
                artifact_type, prediction,
                predictions, entity_predictions, entity_name, subentity_name, img_array, config,
                image_output_dir, img_filename, logger
            )
        except Exception as e:
            logger.error(f"Error creating {artifact_type} artifact for {entity_name} of {subentity_name}: {str(e)}")
            continue
        
        artifacts.append(annotation)
    
    if not artifacts:
        logger.error(f"No artifacts were created successfully")
        raise RuntimeError("No artifacts were created successfully")
    
    # Sample final artifacts with completely non-overlapping patches
    max_artifacts = config['max_artifacts_per_image']

    selected_artifacts = sample_multiple_target_artifacts(
        artifacts,
        max_artifacts=max_artifacts
    )

    if not selected_artifacts:
        logger.error(f"No valid target parts found across all artifact types after sampling")
        raise RuntimeError("No valid target parts found after sampling")
        
    logger.info(f"✅ Successfully selected {len(selected_artifacts)} non-overlapping artifacts")
    total_artifacts = len(selected_artifacts)
    logger.info(f"✅ Created {total_artifacts} artifacts")
    
    return selected_artifacts, visualized_output

def sample_multiple_target_artifacts(
    annotations: List[Dict[str, Any]], 
    max_artifacts: int
) -> List[Dict[str, Any]]:
    """
    Sample multiple target artifacts from annotations with completely non-overlapping patches.
    
    Args:
        annotations: List of artifact annotations
        max_artifacts: Maximum number of artifacts to select
        
    Returns:
        List of selected artifacts with no overlapping patches
    """
    
    # Sort by confidence score (descending order - highest confidence first)
    sorted_annotations = sorted(
        annotations,
        key=lambda x: x.get('sampled_instance_info', {}).get('score', 0),
        reverse=True
    )
    
    selected_artifacts = []
    
    def has_any_patch_overlap(new_artifact: Dict, existing_artifacts: List[Dict]) -> bool:
        """Check if new artifact has any patch overlap with existing ones."""
        new_target_patches = set(new_artifact['target_patch_indices'])
        new_reference_patches = set(new_artifact['reference_patch_indices'])
        
        for existing_artifact in existing_artifacts:
            existing_target_patches = set(existing_artifact['target_patch_indices'])
            existing_reference_patches = set(existing_artifact['reference_patch_indices'])
            
            # Calculate union of all patches for each artifact
            new_all_patches = new_target_patches | new_reference_patches
            existing_all_patches = existing_target_patches | existing_reference_patches
            
            # Check for any overlap between the unions
            if new_all_patches & existing_all_patches:
                return True
                
        return False
    
    # Select artifacts one by one, avoiding any spatial conflicts
    for annotation in sorted_annotations:
        if len(selected_artifacts) >= max_artifacts:
            break
            
        # Check for any patch overlap with already selected artifacts
        if not has_any_patch_overlap(annotation, selected_artifacts):
            selected_artifacts.append(annotation)
    
    return selected_artifacts


def _create_artifact_annotations(
    artifact_type: str,
    prediction: Any,
    predictions: Any,
    entity_predictions: Any,
    entity: str,
    subpart: str,
    img_array: np.ndarray,
    config: Dict[str, Any],
    image_output_dir: str,
    img_filename: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Create artifact annotations for a detected entity-subpart combination.
    
    Args:
        artifact_type: Type of artifact ('addition', 'removal', 'distortion')
        prediction: Prediction data containing mask and bbox information
        predictions: All detection predictions from the model
        entity_predictions: Entity-level detection predictions
        entity: Entity name (e.g., 'person')
        subpart: Subpart name (e.g., 'hand')
        img_array: Image array data
        config: Configuration dictionary with artifact parameters
        image_output_dir: Output directory for this image
        img_filename: Image filename for output files
        logger: Logger instance for tracking progress
        
    Returns:
        Dictionary containing artifact annotation data including bboxes and patch indices
    """
    # Handle random distortion kernel sampling
    distortion_kernel = config['distortion_kernel']
    if config['random_distortion'] and artifact_type == 'distortion':
        available_kernels = ['none', 'jitter', 'swirl', 'voronoi']
        distortion_kernel = random.choice(available_kernels)
        logger.info(f"  Randomly selected distortion kernel: {distortion_kernel}")
    
    # Create artifact patches
    target_patches, reference_patches = InstanceProcessor.create_artifact_patches(
        artifact_type, 
        prediction, 
        predictions, 
        entity_predictions,
        img_array, 
        16, 
        distortion_kernel=distortion_kernel,
        output_dir=image_output_dir,
        img_filename=img_filename
    )

    # Map patches to bounding box coordinates in real image dimensions
    target_bbox = InstanceProcessor.patch_coords_to_bbox(target_patches, patch_size=16)
    reference_bbox = InstanceProcessor.patch_coords_to_bbox(reference_patches, patch_size=16)

    target_mask = InstanceProcessor.patches_to_masks(target_patches, img_array.shape, patch_size=16)
    reference_mask = InstanceProcessor.patches_to_masks(reference_patches, img_array.shape, patch_size=16)
    
    # Create patch mapping
    target_patch_indices, reference_patch_indices = InstanceProcessor.map_coords_to_patch_indices(
        artifact_type=artifact_type,
        target_patches=target_patches,
        reference_patches=reference_patches,
        img_shape=img_array.shape,
        patch_size=16,
        txt_len=512
    )

    logger.info(f"    Target patches: {len(target_patches)} patches -> {len(target_patch_indices)} patch indices")
    logger.info(f"    Reference patches: {len(reference_patches)} patches -> {len(reference_patch_indices)} patch indices")

    # Return structured annotations with artifact index
    return {
        'artifact_type': artifact_type,
        'entity_name': entity,
        'subpart_name': subpart,
        'target_bbox': target_bbox,
        'reference_bbox': reference_bbox,
        'target_patch_indices': target_patch_indices,
        'reference_patch_indices': reference_patch_indices,
        'target_mask': target_mask,
        'reference_mask': reference_mask,
        'distortion_kernel': distortion_kernel if artifact_type == 'distortion' else None
    }


def process_single_image(
    img_info: Dict[str, Any], 
    gsam_detector: GSAMDetector, 
    data_loader: Any,
    visualizer: ImageVisualizer,
    output_dir: str, 
    config: Dict[str, Any], 
    openai_client: openai.OpenAI,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process a single image with GSAM segmentation.
    
    Args:
        img_info: Image information dictionary
        gsam_detector: GSAM detector instance
        data_loader: Data loader instance
        visualizer: Image visualizer instance
        output_dir: Output directory
        config: Configuration dictionary
        openai_client: OpenAI client instance
        logger: Logger instance
        
    Returns:
        Processing results dictionary
    """
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
    image_output_dir = None
    start_time = time.time()
    try:
        # Load image and generate caption using data loader
        if hasattr(data_loader, 'load_image_by_info'):
            img_array = data_loader.load_image_by_info(img_info)
        else:
            img_array = data_loader.load_image_by_path(img_info['file_path'])
        
        caption = data_loader.get_image_caption(img_info)

        # Process all artifact types at once

        try:
            unique_id = str(uuid.uuid4())
            image_output_dir = os.path.join(output_dir, f'{unique_id}')
            os.makedirs(image_output_dir, exist_ok=True)
            (artifacts, visualized_output) = _try_process_all_artifact_types(
                img_array, img_id, gsam_detector, config,
                openai_client, image_output_dir, img_filename, logger
            )
        except Exception as e:
            logger.error(f"Error processing image {img_id}: {str(e)}")
            shutil.rmtree(image_output_dir)
            logger.error(traceback.format_exc())
            raise e

        if not artifacts:
            results['error'] = "No valid target parts found for any artifact type after filtering"
            return results

        # create_visualizations(img_array, img_filename, caption, visualized_output, artifacts, image_output_dir, visualizer)
        real_image_path = visualizer.save_raw_image(img_array, base_dir=image_output_dir, filename="real_image.png")

        unified_data = {
            'id': unique_id,
            'real_image_path': real_image_path,
            'caption': caption,
            'artifacts': artifacts
        }
        # Save unified data
        output_file = os.path.join(image_output_dir, 'metadata.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(unified_data, f)
        
        results['success'] = True
        results['artifacts_created'] = len(artifacts) if artifacts else 0
        
        # Extract artifact types and details from flat list
        artifact_types_processed = list(set(artifact['artifact_type'] for artifact in artifacts))
        results['artifact_types_processed'] = artifact_types_processed
        
        # Group artifact details by type
        artifact_details = {}
        for artifact in artifacts:
            artifact_type = artifact['artifact_type']
            if artifact_type not in artifact_details:
                artifact_details[artifact_type] = []
            artifact_details[artifact_type].append({
                'subpart_name': artifact['subpart_name'], 
            })
        results['artifact_details'] = artifact_details
        logger.info(f"✅ Processed image {img_id} with {results['artifacts_created']} artifacts")

    except Exception as e:
        logger.error(f"❌ Error processing image {img_id}: {str(e)}")
        logger.error(traceback.format_exc())
        results['error'] = str(e)
    
    results['processing_time'] = time.time() - start_time
    return results

def _load_progress_stats(progress_file: str, resume: bool, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load or initialize progress statistics.
    
    Args:
        progress_file: Path to progress file
        resume: Whether to resume from previous run
        logger: Logger instance
        
    Returns:
        Progress statistics dictionary
    """
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
    
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                stats.update(json.load(f))
                logger.info(f"Resuming: {len(stats['processed_image_ids'])} images already processed")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    
    return stats


def _update_progress_stats(
    stats: Dict[str, Any], 
    result: Dict[str, Any], 
    progress_file: str
) -> None:
    """
    Update progress statistics with processing result.
    
    Args:
        stats: Progress statistics dictionary
        result: Processing result for current image
        progress_file: Path to progress file
    """
    stats['processed_images'] += 1
    stats['processed_image_ids'].append(result['image_id'])
    
    if result['success']:
        stats['successful_images'] += 1
        
        # Track artifact type counts (updated for multiple artifacts)
        if 'artifact_types_processed' in result and result['artifact_types_processed']:
            for artifact_type in result['artifact_types_processed']:
                if artifact_type in stats['artifact_counts']:
                    # Count artifacts of each type from artifact_details
                    if 'artifact_details' in result and artifact_type in result['artifact_details']:
                        artifacts_count = len(result['artifact_details'][artifact_type])
                        stats['artifact_counts'][artifact_type] += artifacts_count
        
        # Track detailed artifact statistics
        if 'artifact_details' in result:
            for artifact_type, artifacts_list in result['artifact_details'].items():
                # Initialize detailed stats if not present
                if 'detailed_artifact_stats' not in stats:
                    stats['detailed_artifact_stats'] = {
                        'total_artifacts': 0,
                        'artifacts_per_type': {},
                        'score_distribution': {},
                        'class_distribution': {}
                    }
                
                stats['detailed_artifact_stats']['total_artifacts'] += len(artifacts_list)
                
                # Track artifacts per type
                if artifact_type not in stats['detailed_artifact_stats']['artifacts_per_type']:
                    stats['detailed_artifact_stats']['artifacts_per_type'][artifact_type] = 0
                stats['detailed_artifact_stats']['artifacts_per_type'][artifact_type] += len(artifacts_list)
                
                # Track class distribution
                for artifact in artifacts_list:
                    subpart_name = artifact.get('subpart_name', 'unknown')
                    if subpart_name not in stats['detailed_artifact_stats']['class_distribution']:
                        stats['detailed_artifact_stats']['class_distribution'][subpart_name] = 0
                    stats['detailed_artifact_stats']['class_distribution'][subpart_name] += 1
    else:
        stats['failed_images'] += 1
    
    # Save progress periodically
    if stats['processed_images'] % 10 == 0:
        with open(progress_file, 'w') as f:
            json.dump(stats, f, indent=2)


def _print_processing_summary(
    stats: Dict[str, Any], 
    dataset_type: str, 
    categories: List[str], 
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    Print final processing summary.
    
    Args:
        stats: Progress statistics dictionary
        dataset_type: Type of dataset processed
        categories: List of categories processed
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info("\n" + "="*60)
    logger.info("GSAM PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Processed: {stats['processed_images']}")
    logger.info(f"Successful: {stats['successful_images']}")
    logger.info(f"Failed: {stats['failed_images']}")
    
    success_rate = (stats['successful_images'] / 
                   max(stats['processed_images'], 1) * 100)
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Print artifact type counts
    logger.info("")
    logger.info("ARTIFACT TYPE BREAKDOWN:")
    logger.info("-" * 40)
    for artifact_type, count in stats['artifact_counts'].items():
        percentage = (count / max(stats['successful_images'], 1)) * 100
        logger.info(f"{artifact_type.capitalize()}: {count} artifacts ({percentage:.1f}%)")
    
    # Print detailed artifact statistics if available
    if 'detailed_artifact_stats' in stats:
        detailed_stats = stats['detailed_artifact_stats']
        total_artifacts = detailed_stats.get('total_artifacts', 0)
        
        logger.info("")
        logger.info("DETAILED ARTIFACT STATISTICS:")
        logger.info("-" * 50)
        logger.info(f"Total artifacts created: {total_artifacts}")
        
        if stats['successful_images'] > 0:
            avg_artifacts_per_image = total_artifacts / stats['successful_images']
            logger.info(f"Average artifacts per successful image: {avg_artifacts_per_image:.2f}")
        
        # Show class distribution
        if 'class_distribution' in detailed_stats and detailed_stats['class_distribution']:
            logger.info("")
            logger.info("CLASS DISTRIBUTION:")
            logger.info("-" * 30)
            sorted_classes = sorted(detailed_stats['class_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:  # Show top 10
                percentage = (count / total_artifacts) * 100 if total_artifacts > 0 else 0
                logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            if len(sorted_classes) > 10:
                logger.info(f"  ... and {len(sorted_classes) - 10} more classes")
    
    elapsed_time = (datetime.now() - 
                   datetime.fromisoformat(stats['start_time'])).total_seconds()
    logger.info("")
    logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("="*60)


def run_gsam_processing(
    categories: List[str], 
    max_images: Optional[int] = None, 
    resume: bool = False,
    config: Dict[str, Any] = None,
    openai_client: Optional[openai.OpenAI] = None,
    seed: Optional[int] = None
) -> None:
    """
    Run GSAM segmentation processing on all images.
    
    Args:
        categories: List of categories to process
        max_images: Maximum number of images to process
        resume: Whether to resume from previous run
        config: Configuration dictionary
        openai_client: OpenAI client instance
        seed: Random seed for reproducibility
    """
    # Determine dataset type and setup
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
    
    # Setup and load progress tracking
    progress_file = os.path.join(output_dir, f'processing_progress.json')
    stats = _load_progress_stats(progress_file, resume, logger)
    
    # Initialize components
    logger.info("Initializing components...")
    data_loader = _initialize_data_loader(dataset_type, config)
    
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
    image_list = _get_image_list(dataset_type, data_loader, categories, max_images, logger)
    
    # Shuffle and limit
    if seed is not None:
        random.seed(seed)
    random.shuffle(image_list)
    if max_images:
        image_list = image_list[:max_images]
    
    stats['total_images'] = len(image_list)
    logger.info(f"Processing {len(image_list)} images")
    
    # Filter already processed images if resuming
    if resume:
        processed_ids = set(stats['processed_image_ids'])
        image_list = [img for img in image_list if img['id'] not in processed_ids]
        logger.info(f"Remaining to process: {len(image_list)} images")
    
    if not image_list:
        logger.info("No images to process!")
        return
    
    # Process images with progress tracking
    with tqdm(total=len(image_list), desc=f"Processing {dataset_type} images") as pbar:
        for img_info in image_list:
            result = process_single_image(
                img_info, gsam_detector, data_loader, visualizer,
                output_dir, config, openai_client, logger
            )
            
            # Update statistics and progress
            _update_progress_stats(stats, result, progress_file)
            
            # Update progress bar
            status = "✅" if result['success'] else "❌"
            pbar.set_postfix({
                'Current': f"{status} {result['filename'][:20]}...",
                'Success': stats['successful_images'],
                'Failed': stats['failed_images']
            })
            pbar.update(1)
    
    # Final progress save and summary
    with open(progress_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    _print_processing_summary(stats, dataset_type, categories, output_dir, logger)
    
    # Cleanup components
    gsam_detector.cleanup()


def _create_dataset_config(args: Any) -> Dict[str, Any]:
    """
    Create configuration dictionary based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    base_config = {
        'artifact_types': args.artifact_types,
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
        'distortion_kernel': args.distortion_kernel,
        'random_distortion': args.random_distortion,
        # Artifact configuration
        'max_artifacts_per_image': args.max_artifacts_per_image,
    }
    
    if args.dataset == 'coco':
        dataset_path = args.dataset_path or "../../data/coco_2017_extracted/annotations/"
        image_path = args.image_path or "../../data/coco_2017_extracted/train2017/"
        output_dir = args.output_dir or f'gsam_output_coco_{"-".join(args.categories)}'
        
        base_config.update({
            'dataset_type': 'coco',
            'dataset_path': dataset_path,
            'image_path': image_path,
            'super_categories': args.categories,
            'output_dir': output_dir,
        })
        
    elif args.dataset == 'imagenet':
        dataset_path = args.dataset_path or "../../data/imagenet/"
        output_dir = args.output_dir or f'gsam_output_imagenet_{"-".join(args.categories)}'
        
        base_config.update({
            'dataset_type': 'imagenet',
            'dataset_path': dataset_path,
            'imagenet_split': args.imagenet_split,
            'class_names': args.categories,
            'output_dir': output_dir,
        })
        
    elif args.dataset == 'custom':
        base_config.update({
            'dataset_type': 'custom',
            'dataset_path': args.dataset_path,
            'class_names': args.categories,
            'output_dir': args.output_dir,
        })
    
    return base_config


def _print_startup_info(args: Any, config: Dict[str, Any]) -> None:
    """
    Print startup information about the processing configuration.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary
    """
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
    
    # Artifact configuration info
    print(f"🔢 Max artifacts per image: {args.max_artifacts_per_image}")
    
    if args.random_distortion:
        print(f"🔧 Distortion kernel: random sampling (jitter, swirl, voronoi)")
    else:
        print(f"🔧 Distortion kernel: {args.distortion_kernel}")
    
    print(f"📤 Output directory: {config['output_dir']}")


def _setup_argument_parser() -> Any:
    """
    Setup and return the argument parser for command line interface.
    
    Returns:
        Configured ArgumentParser instance
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate GSAM segmentation results for COCO or ImageNet datasets'
    )
    
    # Required arguments
    parser.add_argument(
        'categories', nargs='+', 
        help='Categories to process. For COCO: supercategories (person, animal, vehicle, etc.). '
             'For ImageNet: class names (dog, cat, car, etc.)'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset', type=str, choices=['coco', 'imagenet', 'custom'], default='coco',
        help='Dataset type to use (default: custom)'
    )
    parser.add_argument(
        '--dataset-path', type=str, default='/home/jhpark/image-artifacts/data/eval_coco_animals',
        help='Path to dataset. For COCO: annotations directory. For ImageNet: root directory '
             'containing train/val folders. For custom: root directory with class subdirectories'
    )
    parser.add_argument(
        '--image-path', type=str, default=None,
        help='Path to images (COCO only). For ImageNet, use --imagenet-split instead'
    )
    parser.add_argument(
        '--imagenet-split', type=str, choices=['train', 'val'], default='train',
        help='ImageNet split to use (train or val)'
    )
    
    # Processing configuration
    parser.add_argument(
        '--artifact-types', nargs='+', 
        default=['distortion', 'removal', 'addition'],
        help='Artifact types to process (only specified types will be generated)'
    )
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Maximum number of images to process'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous run'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    # Filtering parameters
    parser.add_argument(
        '--min-area-ratio', type=float, default=0.005,
        help='Minimum area ratio for part filtering (default: 0.05)'
    )
    parser.add_argument(
        '--max-area-ratio', type=float, default=0.5,
        help='Maximum area ratio for part filtering (default: 0.8)'
    )
    parser.add_argument(
        '--max-ref-overlap', type=float, default=0.3,
        help='Maximum allowed overlap with reference bbox for smart addition (default: 0.3)'
    )
    parser.add_argument(
        '--min-entity-overlap', type=float, default=0.1,
        help='Minimum required overlap with entity bbox for smart addition (default: 0.1)'
    )
    
    # Artifact configuration
    parser.add_argument(
        '--max-artifacts-per-image', type=int, default=3,
        help='Maximum number of artifacts to generate per image (default: 3, set to 1 for single artifact mode)'
    )
    parser.add_argument(
        '--min-score-threshold', type=float, default=0.5,
        help='Minimum score threshold for artifact inclusion (default: 0.5)'
    )
    parser.add_argument(
        '--min-spatial-distance', type=float, default=0.3,
        help='Minimum IoU distance between selected artifacts for spatial diversity (default: 0.3)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', type=str, default='../gsam_output_eval_animals',
        help='Output directory (default: ../gsam_output_eval_animals)'
    )
    parser.add_argument(
        '--predefined-vocab', nargs='+', default=None,
        help='Pre-defined vocabulary list (e.g., --predefined-vocab "person head" "person arm" '
             '"person leg"). If provided, skips OpenAI API calls for vocabulary generation.'
    )
    
    # GSAM model configuration
    parser.add_argument(
        '--grounding-config', type=str, 
        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        help='Path to GroundingDINO config file'
    )
    parser.add_argument(
        '--grounding-checkpoint', type=str, default='weight/groundingdino_swint_ogc.pth',
        help='Path to GroundingDINO checkpoint'
    )
    parser.add_argument(
        '--sam-version', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'],
        help='SAM model version (default: vit_h)'
    )
    parser.add_argument(
        '--sam-checkpoint', type=str, default='weight/sam_vit_h_4b8939.pth',
        help='Path to SAM checkpoint'
    )
    parser.add_argument(
        '--sam-hq-checkpoint', type=str, default=None,
        help='Path to SAM-HQ checkpoint'
    )
    parser.add_argument(
        '--use-sam-hq', action='store_true',
        help='Use SAM-HQ instead of regular SAM'
    )
    
    # Model thresholds
    parser.add_argument(
        '--box-threshold', type=float, default=0.3,
        help='Box threshold for GroundingDINO (default: 0.3)'
    )
    parser.add_argument(
        '--text-threshold', type=float, default=0.25,
        help='Text threshold for GroundingDINO (default: 0.25)'
    )
    parser.add_argument(
        '--nms-threshold', type=float, default=0.5,
        help='NMS threshold for GroundingDINO (default: 0.5)'
    )
    parser.add_argument(
        '--bert-base-uncased-path', type=str, default=None,
        help='Path to BERT base uncased model'
    )
    
    # Distortion configuration
    parser.add_argument(
        '--distortion-kernel', type=str, default='none', 
        choices=['none', 'jitter', 'swirl', 'voronoi'],
        help='Type of distortion kernel to apply for distortion artifacts (default: none)'
    )
    parser.add_argument(
        '--random-distortion', action='store_true',
        help='Randomly sample distortion kernel for each image from jitter, swirl, voronoi'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    return parser


def main() -> None:
    """Main function for GSAM segmentation processing."""
    # Parse arguments
    parser = _setup_argument_parser()
    args = parser.parse_args()

    # Process predefined vocabulary if provided
    if args.predefined_vocab:
        args.predefined_vocab = [vocab.replace('_', ' ') for vocab in args.predefined_vocab]
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Create configuration
    config = _create_dataset_config(args)
    
    # Print startup information
    _print_startup_info(args, config)
    
    # Run processing
    run_gsam_processing(
        categories=args.categories,
        max_images=args.max_images,
        resume=args.resume,
        config=config,
        openai_client=openai.OpenAI(),
        seed=args.seed
    )


def _set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Handle seed setting
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args, _ = parser.parse_known_args()
    
    if args.seed is not None:
        _set_random_seed(args.seed)
    
    main() 