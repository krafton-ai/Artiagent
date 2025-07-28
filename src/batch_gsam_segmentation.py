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
from PIL import Image
import logging
import pickle
import random
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import traceback
from collections import defaultdict

import numpy as np
import torch
import openai

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from pipeline import (
    GSAMDetector, COCODataLoader, ImageNetDataLoader, 
    CustomDirectoryDataLoader, InstanceProcessor, ImageVisualizer
)
from pipeline.prompts import (
    addition_select_candidate, visualize_all_candidates, 
    visualize_candidate_images_for_api, addition_suggest_offset, get_all_entity_subparts
)


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


def create_target_mask_from_patches(
    target_patch_indices: List[int], 
    img_shape: Tuple[int, int, int], 
    patch_size: int = 16
) -> np.ndarray:
    """
    Create target mask from patch indices for addition artifacts.
    
    Args:
        target_patch_indices: List of target patch indices
        img_shape: Image shape (H, W, C)
        patch_size: Size of patches (default 16)
        
    Returns:
        Target mask as numpy array
    """
    from flux.artifacts_util import patch_indices_to_coords
    
    height, width = img_shape[:2]
    patch_width = width // patch_size
    
    # Create empty mask
    target_mask = np.zeros((height, width), dtype=np.uint8)
    
    if target_patch_indices and patch_indices_to_coords is not None:
        # Convert patch indices to coordinates
        patch_coords = patch_indices_to_coords(
            target_patch_indices, patch_width, txt_len=512
        )
        
        # Fill patches in the mask
        for patch_y, patch_x in patch_coords:
            y_start = patch_y * patch_size
            y_end = min((patch_y + 1) * patch_size, height)
            x_start = patch_x * patch_size
            x_end = min((patch_x + 1) * patch_size, width)
            
            target_mask[y_start:y_end, x_start:x_end] = 255
    
    return target_mask


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
        filename="01_original_image.png"
    )
    
    # Save detection results
    visualizer.show_detection_results(
        img_array, 
        visualized_output,
        image_name=img_filename, 
        base_dir=image_output_dir,
        filename="02_detection_results.png"
    )
    
    # Create artifact-specific visualizations
    for artifact_type, artifacts_list in artifacts.items():
        if artifacts_list:  # Check if list is not empty
            # Handle list of artifacts per type
            if not isinstance(artifacts_list, list):
                artifacts_list = [artifacts_list]
            
            # Create combined visualization for all artifacts of this type
            all_target_masks = []
            all_reference_masks = []
            
            for artifact_idx, artifact_data in enumerate(artifacts_list):
                if 'error' not in artifact_data:
                    patch_data = artifact_data.get('patch_data', {})
                    
                    # Create target mask if patch indices exist
                    target_mask = None
                    if 'target_patch_indices' in patch_data:
                        target_mask = create_target_mask_from_patches(
                            patch_data['target_patch_indices'], 
                            img_array.shape, 
                            patch_size=16
                        )
                        if target_mask is not None:
                            all_target_masks.append((target_mask, artifact_idx))
                    
                    # Create reference mask if patch indices exist
                    reference_mask = None
                    if 'reference_patch_indices' in patch_data:
                        reference_mask = create_target_mask_from_patches(
                            patch_data['reference_patch_indices'], 
                            img_array.shape, 
                            patch_size=16
                        )
                        if reference_mask is not None:
                            all_reference_masks.append((reference_mask, artifact_idx))
            
            # Create visualizations for multiple artifacts
            if all_target_masks and all_reference_masks:
                # Create a combined mask structure for visualization
                combined_masks = {}
                
                for idx, (target_mask, artifact_idx) in enumerate(all_target_masks):
                    if idx < len(all_reference_masks):
                        reference_mask, _ = all_reference_masks[idx]
                        
                        # Create artifact-specific key
                        artifact_key = f"{artifact_type}_artifact_{artifact_idx}"
                        combined_masks[artifact_key] = {
                            'reference_mask': reference_mask,
                            'target_mask': target_mask
                        }
                
                # Visualize all artifacts for this type
                if combined_masks:
                    InstanceProcessor.visualize_patch_masks(
                        img_array, 
                        combined_masks, 
                        img_filename, 
                        image_output_dir
                    )


def _try_process_all_artifact_types(
    img_array: np.ndarray,
    img_id: int,
    gsam_detector: GSAMDetector,
    config: Dict[str, Any],
    openai_client: openai.OpenAI,
    output_dir: str,
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
        output_dir: Output directory
        img_filename: Image filename
        logger: Logger instance
        
    Returns:
        Tuple of (artifacts_by_type, visualized_output, combined_vocab)
    """
    logger.info(f"Getting entity subparts for all artifact types...")
    image_output_dir = os.path.join(output_dir, f'image_{img_id}')
    os.makedirs(image_output_dir, exist_ok=True)
    
        # Step 1: Get all entity subparts from API
        
    all_subparts_response = get_all_entity_subparts(openai_client, img_array)
    if not all_subparts_response or 'error' in all_subparts_response:
        logger.info(f"Failed to get entity subparts: {all_subparts_response}")
        shutil.rmtree(image_output_dir)
        return {}, None, None
    
    
    # Transform the response into a nested dictionary structure
    # First level: entity, Second level: subpart, Value: list of artifact types
    entity_subpart_artifacts = defaultdict(lambda: defaultdict(list))
    
    for artifact_type, vocab_data in all_subparts_response.items():
        entity = vocab_data['entity']
        subparts = vocab_data['subparts']
        
        for subpart in subparts:
            entity_subpart_artifacts[entity][subpart].append(artifact_type)
    
    # Step 2: Build combined vocabulary and mappings
    subentity_to_entity = defaultdict(set)  # "ear" -> "giraffe"
    entity_to_artifact_types = defaultdict(set) # "giraffe" -> ["addition", "removal"]
    entities = set()
    subentities = set()
    
    for artifact_type, vocabs in all_subparts_response.items():
            
        entity = vocabs['entity']
        subparts = vocabs['subparts']
        entities.add(entity)
        
        # Track artifact types for this entity
        entity_to_artifact_types[entity].add(artifact_type)
        
        # Build vocabulary and subentity mappings
        for subpart in subparts:
            vocab_item = f"{subpart} of {entity}"
            subentity_to_entity[subpart].add(entity)
            subentities.add(subpart)
        
    logger.info(f"Entities: {list(entities)}")
    logger.info(f"Subentities: {list(subentities)}")

    # Step 3: Single detection call with combined vocabulary
    logger.info(f"Detecting parts using Grounded SAM with combined vocabulary...")
    predictions, entity_predictions, visualized_output = gsam_detector.detect_parts(img_array, list(entities), list(subentities), min_area_ratio=0.001, max_area_ratio=0.5)
    # Step 4: Sample target parts with entity-aware logic
    logger.info(f"Sampling target parts across all entities...")

    artifacts = []
    for artifact_idx, prediction in enumerate(predictions):
        entity_name = prediction['mapped_entity_name']
        subentity_name = prediction['subentity_name']
        
        available_artifact_types = entity_subpart_artifacts.get(entity_name, {}).get(subentity_name, [])
        
        if not available_artifact_types:
            logger.warning(f"No artifact types available for {subentity_name} of {entity_name}, skipping...")
            continue
            
        artifact_type = random.choice(available_artifact_types)
        subpart = prediction['subentity_name']
        mapped_entity_name = prediction['mapped_entity_name']
        sampled_idx = artifact_idx
        
        logger.info(f"Creating {artifact_type} artifact {artifact_idx + 1}: {subpart}")
        
        # Create artifact data for this instance
        annotation = _create_artifact_annotations(
            artifact_type, prediction, sampled_idx,
            predictions, entity_predictions, mapped_entity_name, subpart, img_array, config,
            image_output_dir, img_filename, logger, artifact_idx
        )
        
        artifacts.append(annotation)
    
    if not artifacts:
        logger.info(f"No artifacts were created successfully")
        shutil.rmtree(image_output_dir)
        return [], None
    
    # Sample final artifacts with completely non-overlapping patches
    max_artifacts = config.get('max_artifacts_per_image', 3)

    selected_artifacts = sample_multiple_target_artifacts(
        artifacts,
        max_artifacts=max_artifacts
    )

    if not selected_artifacts:
        logger.info(f"No valid target parts found across all artifact types after sampling")
        shutil.rmtree(image_output_dir)
        return [], None
        
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
    if not annotations:
        return []
    
    # Sort by confidence score (descending order - highest confidence first)
    sorted_annotations = sorted(
        annotations,
        key=lambda x: x.get('sampled_instance_info', {}).get('score', 0),
        reverse=True
    )
    
    selected_artifacts = []
    
    def has_any_patch_overlap(new_artifact: Dict, existing_artifacts: List[Dict]) -> bool:
        """Check if new artifact has any patch overlap with existing ones."""
        new_target_patches = set(new_artifact.get('patch_data', {}).get('target_patch_indices', []))
        new_reference_patches = set(new_artifact.get('patch_data', {}).get('reference_patch_indices', []))
        
        for existing_artifact in existing_artifacts:
            existing_target_patches = set(existing_artifact.get('patch_data', {}).get('target_patch_indices', []))
            existing_reference_patches = set(existing_artifact.get('patch_data', {}).get('reference_patch_indices', []))
            
            # Check for ANY overlap between any combination of patches
            if (new_target_patches & existing_target_patches or          # target-target overlap
                new_reference_patches & existing_reference_patches or    # reference-reference overlap
                new_target_patches & existing_reference_patches or       # new target - existing reference overlap
                new_reference_patches & existing_target_patches):        # new reference - existing target overlap
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
    sampled_idx: int,
    predictions: Any,
    entity_predictions: Any,
    entity: str,
    subpart: str,
    img_array: np.ndarray,
    config: Dict[str, Any],
    image_output_dir: str,
    img_filename: str,
    logger: logging.Logger,
    artifact_idx: int = 0
) -> Dict[str, Any]:
    """
    Create artifact annotations for a successful detection.
    
    Args:
        artifact_type: Type of artifact
        prediction: Prediction data
        sampled_idx: Index of sampled instance
        class_name: Class name
        predictions: Prediction results
        vocab: Vocabulary list
        img_array: Image array
        config: Configuration dictionary
        openai_client: OpenAI client
        image_output_dir: Output directory for this image
        img_filename: Image filename
        logger: Logger instance
        artifact_idx: Index of this artifact among multiple artifacts (default: 0)
        
    Returns:
        Annotations dictionary
    """
    logger.info(f"  Creating artifact data for instance {artifact_idx}...")
    
    # Create masks and patch annotations in one operation
    patch_annotation = InstanceProcessor.create_masks_and_patch_annotations_from_instance(
        prediction, img_array.shape, artifact_type, patch_size=16
    )
    
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
        patch_annotation, 
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
    patch_annotation['target_bbox'] = target_bbox
    patch_annotation['reference_bbox'] = reference_bbox

    # Create patch mapping
    target_patch_indices, reference_patch_indices = InstanceProcessor.patch_coords_to_indices(
        artifact_type=artifact_type,
        target_patches=target_patches,
        reference_patches=reference_patches,
        patch_annot=patch_annotation,
        img_shape=img_array.shape,
        patch_size=16
    )

    # Update patch annotations with mapping
    patch_annotation['target_patch_indices'] = target_patch_indices
    patch_annotation['reference_patch_indices'] = reference_patch_indices
    
    # Create annotation dictionary
    annotation, patch_annotation = InstanceProcessor.create_annotation_dict(
        instance=prediction,
        patch_annot=patch_annotation,
    )
    
    # Return structured annotations with artifact index
    return {
        'artifact_type': artifact_type,
        'artifact_idx': artifact_idx,
        'annotation': annotation,
        'entity_name': entity,
        'subpart_name': subpart,
        'sampled_instance_info': {
            'sampled_idx': sampled_idx,
            'bbox_coords': prediction['pred_box'].cpu().numpy().tolist(),
            'score': prediction['score'].item(),
            'class_idx': prediction['pred_class'].item()
        },
        'patch_data': patch_annotation,
        'distortion_kernel': distortion_kernel if artifact_type == 'distortion' else None
    }


def _convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy arrays and types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object with Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    else:
        return obj


def _save_unified_data(
    img_info: Dict[str, Any],
    img_array: np.ndarray,
    caption: str,
    annotations: List[Dict[str, Any]],  # Now expecting a list of artifacts
    image_output_dir: str
) -> None:
    """
    Save unified data structure to file.
    
    Args:
        img_info: Image information
        img_array: Image array
        caption: Image caption
        annotations: List of artifact dictionaries, each containing artifact_type and other data    
        image_output_dir: Output directory for this image
    """
    # Prepare unified data structure
    unified_data = {
        'image_info': img_info,
        'image_array': img_array,
        'caption': caption,
        'artifacts': []
    }
    # Process each artifact in the annotations list
    for artifact_data in annotations:
        artifact_type = artifact_data['artifact_type']
        unified_artifact = {
            'artifact_type': artifact_type,
            'artifact_name': f"{artifact_data['subpart_name']} of {artifact_data['entity_name']}",
            'target_bbox': artifact_data['patch_data']['target_bbox'],
            'reference_bbox': artifact_data['patch_data']['reference_bbox'],
            'target_patch_indices': artifact_data['patch_data']['target_patch_indices'],
            'reference_patch_indices': artifact_data['patch_data']['reference_patch_indices'],
            'target_mask': artifact_data['annotation']['target_mask'],
            'reference_mask': artifact_data['annotation']['reference_mask'],
            'distortion_kernel': artifact_data['distortion_kernel'] if artifact_type == 'distortion' else None
        }
        unified_data['artifacts'].append(unified_artifact)

    unified_data['processing_timestamp'] = datetime.now().isoformat()
    # Save unified data
    output_file = os.path.join(image_output_dir, 'metadata.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(unified_data, f)


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
    
    start_time = time.time()
    try:
        # Load image and generate caption using data loader
        if hasattr(data_loader, 'load_image_by_info'):
            img_array = data_loader.load_image_by_info(img_info)
        else:
            img_array = data_loader.load_image_by_path(img_info['file_path'])
        
        caption = data_loader.get_image_caption(img_info)

        # Process all artifact types at once
        (artifacts, visualized_output) = _try_process_all_artifact_types(
            img_array, img_id, gsam_detector, config,
            openai_client, output_dir, img_filename, logger
        )

        if not artifacts:
            results['error'] = "No valid target parts found for any artifact type after filtering"
            return results

        # Use artifacts directly (now a flat list of selected artifacts)
        annotations = artifacts

        # Save unified data
        image_output_dir = os.path.join(output_dir, f'image_{img_id}')
        _save_unified_data(
            img_info, img_array, caption, 
            annotations, image_output_dir
        )

        # Create visualizations - group artifacts by type for visualization
        artifacts_for_visualization = {}
        for artifact_data in artifacts:
            artifact_type = artifact_data['artifact_type']
            if artifact_type not in artifacts_for_visualization:
                artifacts_for_visualization[artifact_type] = []
            artifacts_for_visualization[artifact_type].append({
                'annotation': artifact_data['annotation'],
                'patch_data': artifact_data['patch_data']
            })

        create_visualizations( img_array, img_filename, caption, visualized_output,
            artifacts_for_visualization, image_output_dir, visualizer
        )
        
        results['success'] = True
        results['artifacts_created'] = len(annotations) if annotations else 0
        
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
                'score': artifact['sampled_instance_info']['score']
            })
        results['artifact_details'] = artifact_details
        logger.info(f"✅ Processed image {img_id} with {results['artifacts_created']} artifacts")

    except Exception as e:
        logger.error(f"❌ Error processing image {img_id}: {str(e)}")
        logger.error(traceback.format_exc())
        results['error'] = str(e)
    
    results['processing_time'] = time.time() - start_time
    return results


def _initialize_data_loader(dataset_type: str, config: Dict[str, Any]) -> Any:
    """
    Initialize the appropriate data loader based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('coco', 'imagenet', 'custom')
        config: Configuration dictionary
        
    Returns:
        Initialized data loader instance
    """
    if dataset_type == "coco":
        return COCODataLoader(config['dataset_path'], config['image_path'])
    elif dataset_type == "imagenet":
        return ImageNetDataLoader(config['dataset_path'], config['imagenet_split'])
    elif dataset_type == "custom":
        return CustomDirectoryDataLoader(config['dataset_path'])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


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


def _get_coco_image_list(
    data_loader: COCODataLoader, 
    categories: List[str], 
    max_images: Optional[int] = None,
    max_instances_per_image: Optional[int] = 3
) -> List[Dict[str, Any]]:
    """
    Get image list for COCO dataset with optional filtering.
    
    Args:
        data_loader: COCO data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        max_instances_per_image: Maximum instances per image for filtering
        
    Returns:
        List of image information dictionaries
    """
    cat_ids = data_loader.get_category_ids(categories)
    image_list = []
    image_ids_seen = set()
    
    # Count instances per image if filtering is requested
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
                
                if max_images and len(image_list) >= max_images:
                    break
        if max_images and len(image_list) >= max_images:
            break
    
    return image_list


def _get_imagenet_image_list(
    data_loader: ImageNetDataLoader, 
    categories: List[str], 
    max_images: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get image list for ImageNet dataset.
    
    Args:
        data_loader: ImageNet data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        
    Returns:
        List of image information dictionaries
    """
    # Determine target synsets
    target_synsets = []
    for class_name in categories:
        for synset, mapped_name in data_loader.class_mapping.items():
            if mapped_name.lower() == class_name.lower() and synset in data_loader.synsets:
                target_synsets.append(synset)
    
    if not target_synsets:
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
    
    return image_list


def _get_custom_image_list(
    data_loader: CustomDirectoryDataLoader, 
    categories: List[str], 
    max_images: Optional[int] = None,
    logger: logging.Logger = None
) -> List[Dict[str, Any]]:
    """
    Get image list for custom dataset.
    
    Args:
        data_loader: Custom directory data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        logger: Logger instance
        
    Returns:
        List of image information dictionaries
    """
    # Get all available class names from the directory structure
    available_classes = data_loader.get_class_names()
    if logger:
        logger.info(f"Available classes in custom dataset: {available_classes}")
    
    # Filter categories to only include those available in the dataset
    target_classes = [cls for cls in categories if cls in available_classes]
    if not target_classes:
        if logger:
            logger.warning(
                f"None of the specified categories {categories} found in dataset. "
                f"Available: {available_classes}"
            )
        target_classes = available_classes  # Use all available classes
    
    if logger:
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
    
    return image_list


def _get_image_list(
    dataset_type: str, 
    data_loader: Any, 
    categories: List[str], 
    max_images: Optional[int] = None,
    logger: logging.Logger = None
) -> List[Dict[str, Any]]:
    """
    Get image list based on dataset type.
    
    Args:
        dataset_type: Type of dataset
        data_loader: Data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        logger: Logger instance
        
    Returns:
        List of image information dictionaries
    """
    if dataset_type == "coco":
        return _get_coco_image_list(data_loader, categories, max_images)
    elif dataset_type == "imagenet":
        return _get_imagenet_image_list(data_loader, categories, max_images)
    elif dataset_type == "custom":
        return _get_custom_image_list(data_loader, categories, max_images, logger)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


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
    artifact_types: List[str],
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
        artifact_types: List of artifact types to process
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
    logger.info(
        f"Using smart IoU-based bbox generation "
        f"(max_ref_overlap={config['max_ref_overlap']}, "
        f"min_entity_overlap={config['min_entity_overlap']})"
    )
    
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
        'min_score_threshold': args.min_score_threshold,
        'min_spatial_distance': args.min_spatial_distance
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
    print(f"   • Min score threshold: {args.min_score_threshold}")
    print(f"   • Min spatial distance: {args.min_spatial_distance}")
    
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
        artifact_types=args.artifact_types,
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


if __name__ == "__main__":
    # Handle seed setting
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args, _ = parser.parse_known_args()
    
    if args.seed is not None:
        _set_random_seed(args.seed)
    
    main() 