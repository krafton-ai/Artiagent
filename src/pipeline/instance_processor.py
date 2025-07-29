import random
import torch
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flux.artifacts_util import bbox_to_patch_coords, patch_coor_to_ind, mask_to_patch_indices, patch_indices_to_coords, mask_to_patch_coords


class InstanceProcessor:
    """Handler for processing detection instances and generating bbox suggestions"""
    
    @staticmethod
    def filter_instances_by_size(predictions: Dict, min_area_ratio: float = 0.01, max_area_ratio: float = 1.0) -> Tuple[any, torch.Tensor]:
        """
        Filter instances by minimum and maximum area ratio relative to image size
        
        Args:
            predictions: VLPart model predictions
            min_area_ratio: Minimum area ratio relative to total image area
            max_area_ratio: Maximum area ratio relative to total image area
            
        Returns:
            Tuple of (filtered_instances, valid_indices)
        """
        instances = predictions['instances']
        image_height, image_width = instances.image_size
        total_image_area = image_height * image_width
        
        # Calculate areas of all bounding boxes
        boxes = instances.pred_boxes.tensor
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_ratios = areas / total_image_area
        # import ipdb;ipdb.set_trace(context=30)
        # Filter instances that meet both minimum and maximum area requirements
        valid_mask = (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
        
        if valid_mask.sum() == 0:
            print(f"Warning: No instances meet area ratio constraints (min: {min_area_ratio}, max: {max_area_ratio})")
            return None, None
        
        filtered_instances = instances[valid_mask]
        valid_indices = torch.where(valid_mask)[0]
        
        print(f"Filtered {len(instances)} instances to {len(filtered_instances)} (area ratio: {min_area_ratio}-{max_area_ratio})")
        
        return filtered_instances, valid_indices
    
    @staticmethod
    def sample_instance_by_score(predictions: Dict, min_area_ratio: float = 0.01, max_area_ratio: float = 1.0) -> Tuple[Optional[any], Optional[int]]:
        """
        Filter instances by size then sample using scores as weights
        
        Args:
            predictions: VLPart model predictions
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
            
        Returns:
            Tuple of (prediction, original_index) or (None, None) if no valid instances
        """
        filtered_instances, valid_indices = InstanceProcessor.filter_instances_by_size(predictions, min_area_ratio, max_area_ratio)
        
        if filtered_instances is None:
            return None, None
        
        scores = filtered_instances.scores.cpu().numpy()
        
        # # Normalize scores to create probability distribution
        # probabilities = scores / scores.sum()
        
        # # Sample an index based on the probability distribution
        # sampled_idx = random.choices(range(len(filtered_instances)), weights=probabilities, k=1)[0]
        # original_idx = valid_indices[sampled_idx].item()

        sampled_idx = int(np.argmax(scores))
        original_idx = valid_indices[sampled_idx].item()

        return filtered_instances[sampled_idx], original_idx
    
    @staticmethod
    def calculate_iou(box1: Union[List, np.ndarray], box2: Union[List, np.ndarray]) -> float:
        """
        Calculate Intersection over Union of two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [xmin, ymin, xmax, ymax]
            
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0



    @staticmethod
    def generate_candidate_addition_region(reference_instance, predictions, mask_patch_coords, img_shape, patch_size: int = 16, overlap_threshold: float = 0.05) -> Tuple[Optional[List], Optional[Dict]]:
        """
        Generate sophisticated bbox suggestion for addition artifacts using IoU calculations
        
        Args:
            predictions: VLPart model predictions
            reference_instance: The reference instance
            reference_class_name: Name of the reference class (e.g., "person hand")
            vocab: List of class names
            img_shape: Image shape (height, width, channels)
            artifact_direction: Direction for artifact placement
            max_ref_overlap: Maximum allowed overlap with reference bbox (default: 0.3)
            min_entity_overlap: Minimum required overlap with entity bbox (default: 0.1)
            
        Returns:
            Tuple of (candidate_list, metadata_dict) or (None, None) if no valid suggestion
        """        
        H, W = img_shape[:2]
        
        # Get reference bbox and mask
        ref_bbox = reference_instance['pred_box'].cpu().numpy()
        ref_mask = reference_instance['pred_mask'].cpu().numpy()
        
        # Step 1: Choose entity (vocab[0]) with highest overlap with reference_instance
        entity_instances = []
        
        for i, pred_instance in enumerate(predictions):
            if pred_instance['pred_class'] == 0:
                entity_instances.append({
                    'idx': i,
                    'bbox': pred_instance['pred_box'].cpu().numpy(),
                    'mask': pred_instance['pred_mask'].cpu().numpy()
                })
        
        if not entity_instances:
            raise ValueError("No entity instances found")
        
        # Find entity with highest overlap with reference
        best_entity = None
        max_overlap = 0.9
        
        for entity in entity_instances:
            # Calculate mask overlap using patch-aligned masks
            entity_mask_patch = InstanceProcessor._align_mask_to_patches(entity['mask'].astype(np.uint8), patch_size)
            intersection = np.sum(ref_mask_patch & entity_mask_patch)
            ref_area = np.sum(ref_mask_patch)
            overlap = intersection / ref_area if ref_area > 0 else 0.0
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_entity = entity
        
        if best_entity is None:
            raise ValueError("No overlapping entity found")
        
        # Step 2: Get instances that are not reference_instance but have same class_name
        reference_class_idx = reference_instance['pred_class'].item()
        
        same_class_instances = []
        for i, pred_instance in enumerate(predictions):
            if (pred_instance['pred_class'] == reference_class_idx and 
                not torch.equal(pred_instance['pred_box'], torch.from_numpy(ref_bbox).float())):
                same_class_instances.append({
                    'idx': i,
                    'bbox': pred_instance['pred_box'].cpu().numpy(),
                    'mask': pred_instance['pred_mask'].cpu().numpy()
                })
        
        # Step 3: Generate stratified angles
        def generate_stratified_offsets(n_bins=6):
            angles = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
            offsets = []
            for angle in angles:
                dx = np.cos(angle)
                dy = np.sin(angle)
                offsets.append((dx, dy))
            return offsets
        
        stratified_offsets = generate_stratified_offsets(8)
        
        # Get reference dimensions
        ref_h = ref_bbox[3] - ref_bbox[1]
        ref_w = ref_bbox[2] - ref_bbox[0]
        
        # Step 4: For each radius, test candidate positions
        radii = [0.8, 1.0, 1.2]  # Multiples of reference size
        candidates = []
        
        for dx, dy in stratified_offsets:
            for radius in radii:
                
                # Step 4-1: Calculate offset
                offset_x = dx * radius * ref_w
                offset_y = dy * radius * ref_h
                
                # Step 4-2: Get target mask by shifting reference mask
                offset_x_int = int(round(offset_x))
                offset_y_int = int(round(offset_y))
                
                # Create target mask by shifting reference mask
                target_mask = np.zeros((H, W), dtype=np.uint8)
                
                # Find reference mask pixels
                ref_y_coords, ref_x_coords = np.where(ref_mask > 0)
                
                # Apply offset to coordinates
                target_y_coords = ref_y_coords + offset_y_int
                target_x_coords = ref_x_coords + offset_x_int
                
                # Filter coordinates that are within image bounds
                valid_mask = ((target_y_coords >= 0) & (target_y_coords < H) & 
                             (target_x_coords >= 0) & (target_x_coords < W))
                
                valid_target_y = target_y_coords[valid_mask]
                valid_target_x = target_x_coords[valid_mask]
                
                # Skip if too few pixels remain after translation
                if len(valid_target_y) < 0.5 * np.sum(ref_mask > 0):
                    continue
                
                # Set target mask pixels
                target_mask[valid_target_y, valid_target_x] = 1
                
                # Calculate target bbox from the actual mask for metadata
                if np.sum(target_mask > 0) > 0:
                    y_indices, x_indices = np.where(target_mask > 0)
                    target_bbox = np.array([
                        np.min(x_indices),  # xmin
                        np.min(y_indices),  # ymin
                        np.max(x_indices),  # xmax
                        np.max(y_indices)   # ymax
                    ], dtype=np.float32)
                else:
                    continue
                
                # Step 4-3: Calculate three thresholds
                
                # Step 4-3-1: Overlap with entity
                intersection = np.sum(target_mask & (best_entity['mask'] > 0))
                target_area = np.sum(target_mask > 0)
                entity_overlap = intersection / target_area if target_area > 0 else 0.0
                
                # Step 4-3-2: Inter-instance overlap (with same class instances)
                max_inter_overlap = 0.0
                for same_inst in same_class_instances:
                    intersection = np.sum(target_mask & (same_inst['mask'] > 0))
                    target_area = np.sum(target_mask > 0)
                    overlap = intersection / target_area if target_area > 0 else 0.0
                    max_inter_overlap = max(max_inter_overlap, overlap)
                
                # Step 4-3-3: Intra-instance overlap (with reference)
                intersection = np.sum(target_mask & (ref_mask > 0))
                target_area = np.sum(target_mask > 0)
                intra_overlap = intersection / target_area if target_area > 0 else 0.0
                
                # Step 4-3-4: Check thresholds
                if (entity_overlap >= overlap_threshold and 
                    max_inter_overlap <= overlap_threshold and 
                    intra_overlap <= overlap_threshold):
                    
                    candidate = {
                        'target_mask': target_mask,
                        'target_bbox': target_bbox,
                        'offset': (offset_x, offset_y),
                        'entity_overlap': entity_overlap,
                        'inter_overlap': max_inter_overlap,
                        'intra_overlap': intra_overlap,
                        'radius': radius,
                        'angle': np.arctan2(dy, dx)
                    }
                    candidates.append(candidate)
                    break  # Break from radius loop for this angle
        
        # Step 5: Return list of candidates
        if not candidates:
            raise ValueError("No valid candidate regions found")
        
        metadata = {
            'best_entity': best_entity,
            'same_class_count': len(same_class_instances),
            'tested_positions': len(radii) * len(stratified_offsets)
        }
        
        return candidates, metadata
    
    @staticmethod
    def create_artifact_patches(artifact_type: str, prediction: Dict, predictions: Dict, entity_predictions: Dict, img_array, patch_size: int = 16, distortion_kernel: str = 'none', output_dir: str = None, img_filename: str = None) -> Tuple[List[int], List[int]]:
        """
        Create artifact patches for different artifact types
        """
        H, W = img_array.shape[:2]
        patch_H = (H // patch_size) * patch_size
        patch_W = (W // patch_size) * patch_size
        patch_w = patch_W // patch_size
        patch_h = patch_H // patch_size

        mask = prediction['pred_mask'].cpu().numpy()
        mask_patch_coords = mask_to_patch_coords(mask, patch_size=patch_size)

        if artifact_type == 'addition':
            offset, prob_map, metadata = InstanceProcessor.generate_addition_probability_map(prediction, predictions, entity_predictions, mask_patch_coords, img_array.shape, patch_size=patch_size, alpha=2.0, max_entity_overlap=0.7, distance_penalty_weight=0.05)
            InstanceProcessor.visualize_addition_probability_map(img_array, prob_map, prediction, metadata, patch_size=patch_size, output_dir=output_dir, img_filename=img_filename)

            offset_x, offset_y = offset
            shift_x_patches = int(offset_x / patch_size)
            shift_y_patches = int(offset_y / patch_size)
            
            # Calculate shifted target patch coordinates
            target_patches = []
            for mask_py, mask_px in mask_patch_coords:
                shifted_py = mask_py + shift_y_patches
                shifted_px = mask_px + shift_x_patches
                target_patches.append((shifted_py, shifted_px))
            reference_patches = mask_patch_coords
        elif artifact_type == 'removal':
            reference_patches = mask_patch_coords
            # Find surrounding patches within Hamilton distance of 3
            surrounding_patches = []
            ref_patch_set = set(mask_patch_coords)  # Convert to set for fast lookup
            
            for ref_py, ref_px in mask_patch_coords:
                for dy in range(-1, 2):  # -3 to 3
                    for dx in range(-1, 2):
                        if abs(dy) + abs(dx) <= 1:  # Hamilton distance <= 3
                            surr_py = ref_py + dy
                            surr_px = ref_px + dx
                            # Only add if within bounds AND not in reference patches
                            if (0 <= surr_py < patch_h and 0 <= surr_px < patch_w and 
                                (surr_py, surr_px) not in ref_patch_set):
                                surrounding_patches.append((surr_py, surr_px))
            
            # Remove duplicates
            surrounding_patches = list(set(surrounding_patches))
            # Get reference class from sampled instance
            reference_class_idx = prediction['pred_class'].item()
            # Pre-compute set of patches that contain ANY prediction instance pixels (foreground patches)
            foreground_patches = set()
            for entity_pred_instance in entity_predictions:
                if prediction['mapped_entity_name'] == entity_pred_instance['entity_name']:
                    entity_mask = entity_pred_instance['pred_mask'].cpu().numpy()
                    entity_patch_indices = mask_to_patch_indices(entity_mask, patch_size=patch_size, txt_len=512)
                    entity_patch_coords = patch_indices_to_coords(entity_patch_indices, patch_w, txt_len=512)
                    entity_patch_coords = [tuple(coord) for coord in entity_patch_coords]
                    foreground_patches.update(entity_patch_coords)
            # Pre-compute set of patches that contain same-class-different-instance pixels
            conflicting_patches = set()
            for pred_instance in predictions:
                if pred_instance['pred_class'] == reference_class_idx and not torch.equal(pred_instance['pred_box'], prediction['pred_box']):
                    instance_mask = pred_instance['pred_mask'].cpu().numpy()
                    instance_patch_indices = mask_to_patch_indices(instance_mask, patch_size=patch_size, txt_len=512)
                    instance_patch_coords = patch_indices_to_coords(instance_patch_indices, patch_w, txt_len=512)
                    instance_patch_coords = [tuple(coord) for coord in instance_patch_coords]
                    conflicting_patches.update(instance_patch_coords)

            surrounding_patches_set = set(surrounding_patches)
            filtered_surrounding_patches = list(surrounding_patches_set - conflicting_patches)
            # Get intersection of foreground patches and filtered surrounding patches
            non_intersecting_patches = list(surrounding_patches_set - foreground_patches)
            if len(non_intersecting_patches) > len(filtered_surrounding_patches) // 2:
                filtered_surrounding_patches = non_intersecting_patches
            if len(filtered_surrounding_patches) == 0:
                raise ValueError("No valid surrounding patches found")
            
            target_patches = mask_patch_coords
            reference_patches = filtered_surrounding_patches
        elif artifact_type == 'distortion':
            target_patches = mask_patch_coords
            
            # Apply distortion kernel based on specified type
            if distortion_kernel == 'none':
                reference_patches = []
            elif distortion_kernel == 'jitter':
                # Apply Gaussian jitter kernel
                reference_patches = InstanceProcessor.gaussian_jitter_kernel(
                    mask_patch_coords, sigma=1.0, patch_h=patch_h, patch_w=patch_w
                )
            elif distortion_kernel == 'swirl':
                # Apply swirl kernel
                reference_patches = InstanceProcessor.swirl_kernel(
                    mask_patch_coords, strength=0.5, patch_h=patch_h, patch_w=patch_w
                )
            elif distortion_kernel == 'voronoi':
                # Apply Voronoi seed kernel
                reference_patches = InstanceProcessor.voronoi_seed_kernel(
                    mask_patch_coords, seed_fraction=0.15, patch_h=patch_h, patch_w=patch_w
                )
            elif distortion_kernel == 'flip':
                # Apply flip kernel
                reference_patches = InstanceProcessor.flip_kernel(
                    mask_patch_coords, patch_h=patch_h, patch_w=patch_w
                )
            else:
                raise ValueError(f"Unknown distortion kernel: {distortion_kernel}")
            
            print(f"Applied {distortion_kernel} distortion kernel: {len(mask_patch_coords)} -> {len(reference_patches)} patches")
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        return target_patches, reference_patches
    
    @staticmethod
    def map_coords_to_patch_indices(artifact_type: str, target_patches: List[Tuple[int, int]], reference_patches: List[Tuple[int, int]],
                           img_shape: Tuple[int, int], 
                           patch_size: int = 16,
                           txt_len: int = 512) -> Dict:
        """
        Create patch mapping for different artifact types
        
        Args:
            artifact_type: Type of artifact ('addition', 'removal', 'distortion')
            target_patches: List of target patch coordinates
            reference_patches: List of reference patch coordinates
            img_shape: Image shape (H, W)
            patch_size: Size of patches (default 16 for FLUX)
            
        Returns:
            Dictionary containing target_patch_indices and reference_patch_indices
        """
        H, W = img_shape[:2]
        patch_H = (H // patch_size) * patch_size
        patch_W = (W // patch_size) * patch_size
        patch_w = patch_W // patch_size
        patch_h = patch_H // patch_size

        if artifact_type == 'addition':
            # Create feasibility mask
            feasibility_mask = []
            for target_py, target_px in target_patches:
                is_feasible = (0 <= target_py < patch_h and 0 <= target_px < patch_w)
                feasibility_mask.append(is_feasible)
            
            # Generate one-to-one mapping of feasible patches
            reference_patch_indices = [patch_coor_to_ind(ref_px, ref_py, patch_w, txt_len) for ref_py, ref_px in reference_patches]
            target_patch_indices = []
            feasible_reference_patch_indices = []
            
            for i, is_feasible in enumerate(feasibility_mask):
                if is_feasible:
                    ref_patch_idx = reference_patch_indices[i]
                    feasible_reference_patch_indices.append(ref_patch_idx)
                    target_py, target_px = target_patches[i]
                    
                    target_patch_idx = patch_coor_to_ind(target_px, target_py, patch_w, txt_len)
                    target_patch_indices.append(target_patch_idx)
            
            return target_patch_indices, feasible_reference_patch_indices

        elif artifact_type == 'removal':
            reference_patch_indices = [patch_coor_to_ind(ref_px, ref_py, patch_w, txt_len) for ref_py, ref_px in reference_patches]
            target_patch_indices = [patch_coor_to_ind(tar_px, tar_py, patch_w, txt_len) for tar_py, tar_px in target_patches]
            return target_patch_indices, reference_patch_indices
        
        elif artifact_type == 'distortion':
            reference_patch_indices = [patch_coor_to_ind(ref_px, ref_py, patch_w, txt_len) for ref_py, ref_px in reference_patches]
            target_patch_indices = [patch_coor_to_ind(tar_px, tar_py, patch_w, txt_len) for tar_py, tar_px in target_patches]
            return target_patch_indices, reference_patch_indices
        
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

    @staticmethod
    def patch_coords_to_bbox(patch_coords: List[Tuple[int, int]], patch_size: int = 16) -> Tuple[int, int, int, int]:
        """
        Convert patch coordinates to bounding box coordinates in real image dimensions.
        
        Args:
            patch_coords: List of tuples (py, px) representing patch coordinates
            patch_size: Size of each patch in pixels (default 16)
            
        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in pixel coordinates
        """
        if not patch_coords:
            return (0, 0, 0, 0)
        
        # Extract patch coordinates
        patch_ys, patch_xs = zip(*patch_coords)
        
        # Convert to pixel coordinates
        min_patch_y, max_patch_y = min(patch_ys), max(patch_ys)
        min_patch_x, max_patch_x = min(patch_xs), max(patch_xs)
        
        # Calculate bounding box in pixel coordinates
        xmin = min_patch_x * patch_size
        ymin = min_patch_y * patch_size
        xmax = (max_patch_x + 1) * patch_size  # +1 because we want to include the entire patch
        ymax = (max_patch_y + 1) * patch_size
        
        return (xmin, ymin, xmax, ymax)

    @staticmethod
    def patches_to_masks(patch_coords, img_shape, patch_size: int = 16) -> np.ndarray:
        """
        Convert patch coordinates to binary mask.
        
        Args:
            patch_coords: List of tuples (py, px) representing patch coordinates
            img_shape: Shape of the image (H, W, C) or (H, W)
            patch_size: Size of each patch in pixels (default 16)
            
        Returns:
            Binary mask as numpy array with same height/width as image.
            Pixels covered by patches are set to 1, others to 0.
        """
        
        img_height, img_width = img_shape[:2]
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for py, px in patch_coords: 
            y_start = py * patch_size
            y_end = min((py + 1) * patch_size, img_height)
            x_start = px * patch_size
            x_end = min((px + 1) * patch_size, img_width)
            mask[y_start:y_end, x_start:x_end] = 1
        
        return mask

    @staticmethod
    def create_addition_patch_mapping(annotation, img_shape, reference_patch_indices, patch_size=16):
        """
        Create one-to-one patch mapping for addition artifacts using two-list algorithm
        
        Args:
            annotation: Annotation dictionary with bounding_box (target location)
            img_shape: Shape of the image (H, W)
            reference_patch_indices: List of reference patch indices
            patch_size: Size of patches (default 16 for FLUX)
            
        Returns:
            Dictionary containing target patch indices, one-to-one mapping, and truncated reference patch indices.
        """
        H, W = img_shape[:2]
        patch_H = (H // patch_size) * patch_size
        patch_W = (W // patch_size) * patch_size
        patch_w = patch_W // patch_size
        patch_h = patch_H // patch_size
        
        # Get bounding box coordinates  
        bbox = annotation['bounding_box']
        ref_bbox = annotation['bounding_box_ref']
        
        # Convert ALL reference patch indices to coordinates
        ref_patch_coords = patch_indices_to_coords(reference_patch_indices, patch_w, txt_len=512)
        
        if not ref_patch_coords:
            return {'target_patch_indices': [], 'reference_patch_indices': []}
        
        # Use predefined offset from bbox suggestion
        offset_x, offset_y = annotation['offset']
    
        # Convert offset to patch coordinates
        shift_x_patches = int(offset_x / patch_size)
        shift_y_patches = int(offset_y / patch_size)
        
        # First list: shifted target patch coordinates (same length as reference)
        shifted_target_coords = []
        for ref_py, ref_px in ref_patch_coords:
            shifted_py = ref_py + shift_y_patches
            shifted_px = ref_px + shift_x_patches
            shifted_target_coords.append((shifted_py, shifted_px))
        
        # Second list: feasibility mask (boolean array, same length as reference)
        feasibility_mask = []
        for shifted_py, shifted_px in shifted_target_coords:
            is_feasible = (0 <= shifted_py < patch_h and 0 <= shifted_px < patch_w)
            feasibility_mask.append(is_feasible)
        
        # Use the mask to generate one-to-one mapping of feasible patches
        target_patch_indices = []
        feasible_reference_patch_indices = []
        
        for i, is_feasible in enumerate(feasibility_mask):
            if is_feasible:
                ref_patch_idx = reference_patch_indices[i]
                feasible_reference_patch_indices.append(ref_patch_idx)
                shifted_py, shifted_px = shifted_target_coords[i]
                
                target_patch_idx = patch_coor_to_ind(shifted_px, shifted_py, patch_w, txt_len=512)
                target_patch_indices.append(target_patch_idx)        
        return {
            'target_patch_indices': target_patch_indices,
            'reference_patch_indices': feasible_reference_patch_indices,
        }

    @staticmethod
    def create_patch_aware_segmentation_mask(instance, img_shape, patch_size: int = 16) -> np.ndarray:
        """
        Create reference mask aligned to patch boundaries
        
        Args:
            instance: VLPart instance with bbox and potentially mask
            img_shape: Shape of the image (H, W)
            patch_size: Size of patches (default 16 for FLUX)
            
        Returns:
            Reference mask aligned to patch boundaries
        """
        H, W = img_shape[:2]
        
        # Ensure dimensions are compatible with patch size
        patch_H = (H // patch_size) * patch_size
        patch_W = (W // patch_size) * patch_size
        
        # Get bounding box coordinates
        bbox_coords = instance.pred_boxes.tensor[0].cpu().numpy()
        xmin, ymin, xmax, ymax = bbox_coords.astype(int)
        
        # Adjust to patch-aligned dimensions
        xmin = max(0, min(xmin, patch_W))
        xmax = max(0, min(xmax, patch_W))
        ymin = max(0, min(ymin, patch_H))
        ymax = max(0, min(ymax, patch_H))
        
        # Check if instance has segmentation mask
        if hasattr(instance, 'pred_masks') and instance.pred_masks is not None:
            # Use the actual segmentation mask but align to patches
            seg_mask = instance.pred_masks[0].cpu().numpy().astype(np.uint8)
            
            # Crop to patch-aligned dimensions
            seg_mask_cropped = seg_mask[:patch_H, :patch_W]
            
            # Convert to patch-aligned mask
            reference_mask = InstanceProcessor._align_mask_to_patches(seg_mask_cropped, patch_size)
        else:
            # Create reference mask from bounding box at patch granularity
            adjusted_bbox = (xmin, xmax, ymin, ymax)
            if bbox_to_patch_coords is not None:
                patch_coords = bbox_to_patch_coords(adjusted_bbox, patch_size=patch_size)
            else:
                patch_coords = []
            
            reference_mask = np.zeros((patch_H, patch_W), dtype=np.uint8)
            for py, px in patch_coords:
                y_start = py * patch_size
                y_end = min((py + 1) * patch_size, patch_H)
                x_start = px * patch_size
                x_end = min((px + 1) * patch_size, patch_W)
                reference_mask[y_start:y_end, x_start:x_end] = 1
        
        # If original image was larger than patch-aligned dimensions, pad the mask
        if H > patch_H or W > patch_W:
            padded_mask = np.zeros((H, W), dtype=np.uint8)
            padded_mask[:patch_H, :patch_W] = reference_mask
            reference_mask = padded_mask
        
        return reference_mask

    @staticmethod
    def _align_mask_to_patches(mask: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """
        Align a mask to patch boundaries
        
        Args:
            mask: Input mask
            patch_size: Size of patches
            
        Returns:
            Mask aligned to patch boundaries
        """
        H, W = mask.shape
        patch_H, patch_W = H // patch_size, W // patch_size
        
        aligned_mask = np.zeros((patch_H * patch_size, patch_W * patch_size), dtype=np.uint8)
        
        for py in range(patch_H):
            for px in range(patch_W):
                # Get the patch region in the original mask
                y_start, y_end = py * patch_size, (py + 1) * patch_size
                x_start, x_end = px * patch_size, (px + 1) * patch_size
                
                patch_region = mask[y_start:y_end, x_start:x_end]
                
                # If any part of the patch contains mask pixels, fill the entire patch
                if np.any(patch_region > 0):
                    aligned_mask[y_start:y_end, x_start:x_end] = 1
        
        return aligned_mask

    @staticmethod
    def visualize_patch_masks(img_array: np.ndarray, masks_data: Dict, 
                            img_filename: str, output_dir: str, patch_size: int = 16):
        """
        Create visualizations showing the patch-based masks
        
        Args:
            img_array: Source image array
            masks_data: Dictionary containing masks for each artifact type
            img_filename: Image filename for output directory
            output_dir: Base output directory
            patch_size: Size of patches
        """

        
        # Use output_dir directly (no additional subdirectory creation)
        viz_output_dir = output_dir
        os.makedirs(viz_output_dir, exist_ok=True)
        
        for artifact_type, masks in masks_data.items():
            if 'error' in masks:
                continue
                
            reference_mask = masks.get('reference_mask')
            target_mask = masks.get('target_mask')
            
            if artifact_type == 'addition' and target_mask is not None:
                # Show 3 panels for addition: original, reference, target
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img_array)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Reference mask with patch grid
                axes[1].imshow(img_array)
                if reference_mask is not None:
                    axes[1].imshow(reference_mask, alpha=0.5, cmap='Reds')
                InstanceProcessor._add_patch_grid(axes[1], img_array.shape, patch_size)
                axes[1].set_title(f'Reference Mask ({artifact_type})')
                axes[1].axis('off')
                
                # Target mask with patch grid
                axes[2].imshow(img_array)
                axes[2].imshow(target_mask, alpha=0.5, cmap='Blues')
                InstanceProcessor._add_patch_grid(axes[2], img_array.shape, patch_size)
                axes[2].set_title(f'Target Mask ({artifact_type})')
                axes[2].axis('off')
                
            elif reference_mask is not None:
                # Show 2 panels for removal/distortion: original, reference
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Original image
                axes[0].imshow(img_array)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Reference mask with patch grid
                axes[1].imshow(img_array)
                axes[1].imshow(reference_mask, alpha=0.5, cmap='Reds')
                InstanceProcessor._add_patch_grid(axes[1], img_array.shape, patch_size)
                axes[1].set_title(f'Reference Mask ({artifact_type})')
                axes[1].axis('off')
            else:
                continue
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_output_dir, f"03_patch_masks_{artifact_type}.png"), 
                        dpi=150, bbox_inches='tight')
            plt.close()
            

    @staticmethod
    def _add_patch_grid(ax, img_shape, patch_size: int = 16, alpha: float = 0.7):
        """Add patch grid overlay to visualization"""
        try:
            H, W = img_shape[:2]
            
            # Add vertical lines
            for x in range(0, W, patch_size):
                ax.axvline(x=x, color='white', linewidth=0.5, alpha=alpha)
            
            # Add horizontal lines  
            for y in range(0, H, patch_size):
                ax.axhline(y=y, color='white', linewidth=0.5, alpha=alpha)
                
        except Exception:
            # Silently fail if visualization cannot be created
            pass 

    # ---- Distortion Kernel Functions ----
    @staticmethod
    def gaussian_jitter_kernel(
        mask_patch_coords: List[Tuple[int, int]],
        sigma: float = 1.0,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Apply i.i.d. Gaussian jitter to each patch coordinate.

        Args:
            mask_patch_coords: List of (py, px) integers to be distorted.
            sigma: Standard deviation of the normal noise added to each axis, in patch units.
            patch_h, patch_w: Optional explicit patch‑grid height/width.  If not given they
                are inferred from the maximum coordinates in `mask_patch_coords`.

        Returns:
            A list of jittered (py, px) coordinates, one‑to‑one with the input order.
        """
        if not mask_patch_coords:
            return []

        if patch_h is None or patch_w is None:
            max_py = max(py for py, _ in mask_patch_coords)
            max_px = max(px for _, px in mask_patch_coords)
            patch_h = max_py + 1
            patch_w = max_px + 1

        jittered = []
        for py, px in mask_patch_coords:
            n_py = int(round(py + np.random.normal(0, sigma)))
            n_px = int(round(px + np.random.normal(0, sigma)))
            n_py = max(0, min(n_py, patch_h - 1))
            n_px = max(0, min(n_px, patch_w - 1))
            jittered.append((n_py, n_px))
        return jittered

    @staticmethod
    def swirl_kernel(
        mask_patch_coords: List[Tuple[int, int]],
        strength: float = 0.5,
        center: Optional[Tuple[float, float]] = None,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Discrete swirl distortion: rotate each coordinate around a centre by an amount
        proportional to its distance from the centre.

        Args:
            mask_patch_coords: List of (py, px) to distort.
            strength: Angular coefficient in radians per normalised radius.
            center: (cy, cx) in patch units.  Defaults to centroid of coords.
            patch_h, patch_w: Optional patch‑grid dimensions for clamping.

        Returns:
            List of (py, px) after swirl, aligned 1‑to‑1 with input.
        """
        if not mask_patch_coords:
            return []

        if center is None:
            cy = sum(py for py, _ in mask_patch_coords) / len(mask_patch_coords)
            cx = sum(px for _, px in mask_patch_coords) / len(mask_patch_coords)
        else:
            cy, cx = center

        if patch_h is None or patch_w is None:
            max_py = max(py for py, _ in mask_patch_coords)
            max_px = max(px for _, px in mask_patch_coords)
            patch_h = max_py + 1
            patch_w = max_px + 1

        # Maximum radius for normalisation
        max_r = max(
            np.hypot(py - cy, px - cx) for py, px in mask_patch_coords
        ) + 1e-6  # avoid div‑by‑zero

        new_coords = []
        for py, px in mask_patch_coords:
            dy, dx = py - cy, px - cx
            r = np.hypot(dy, dx)
            theta = np.arctan2(dy, dx) + strength / max(0.1, (r / max_r))
            n_py = int(round(cy + r * np.sin(theta)))
            n_px = int(round(cx + r * np.cos(theta)))
            n_py = max(0, min(n_py, patch_h - 1))
            n_px = max(0, min(n_px, patch_w - 1))
            new_coords.append((n_py, n_px))
        return new_coords

    @staticmethod
    def voronoi_seed_kernel(
        mask_patch_coords: List[Tuple[int, int]],
        seed_fraction: float = 0.15,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int]]:
        """
        Duplicate embeddings via a Voronoi remap: sample seed patches and assign every
        other patch to its nearest seed.

        Args:
            mask_patch_coords: Coordinates to remap.
            seed_fraction: Fraction (0,1] of coords used as seeds. At least one seed always.
            patch_h, patch_w: Explicit grid size for clamping (inferred if None).
            rng: Optional `random.Random` instance for reproducibility.

        Returns:
            List where each entry is the coordinate of the seed that the corresponding
            input patch copies.
        """
        if not mask_patch_coords:
            return []

        if rng is None:
            rng = random

        n_coords = len(mask_patch_coords)
        n_seeds = max(1, int(round(seed_fraction * n_coords)))
        seeds = rng.sample(mask_patch_coords, n_seeds)

        # Pre‑compute squared distances for efficiency
        seeds_arr = np.array(seeds)
        coords_arr = np.array(mask_patch_coords)
        dists = ((coords_arr[:, None, :] - seeds_arr[None, :, :]) ** 2).sum(-1)
        nearest_idx = dists.argmin(axis=1)

        new_coords = [tuple(seeds[idx]) for idx in nearest_idx]
        return new_coords
    

    @staticmethod
    def flip_kernel(
        mask_patch_coords: List[Tuple[int, int]],
        direction: Optional[str] = None,  # Unused in new logic
        patch_h: Optional[int] = None,    # Unused in new logic
        patch_w: Optional[int] = None,    # Unused in new logic
    ) -> List[Tuple[int, int]]:
        """
        Flip patch coordinates by mirroring across the centroid, always mapping to a coordinate within the region.
        If the mirrored coordinate is not in the region, use the nearest coordinate in the region.
        Args:
            mask_patch_coords: List of (py, px) coordinates to be flipped.
            direction: Ignored in this version; always mirrors across centroid.
        Returns:
            List of flipped (py, px) coordinates, one-to-one with input order, always within the region.
        """
        if not mask_patch_coords:
            return []

        coords_arr = np.array(mask_patch_coords)
        centroid_y = np.mean(coords_arr[:, 0])
        centroid_x = np.mean(coords_arr[:, 1])
        region_set = set(mask_patch_coords)
        flipped = []
        for py, px in mask_patch_coords:
            # Mirror across centroid
            mirrored_y = int(round(2 * centroid_y - py))
            mirrored_x = int(round(2 * centroid_x - px))
            candidate = (mirrored_y, mirrored_x)
            if candidate in region_set:
                flipped.append(candidate)
            else:
                # Find nearest coordinate in region
                dists = np.sum((coords_arr - np.array([mirrored_y, mirrored_x])) ** 2, axis=1)
                nearest_idx = np.argmin(dists)
                flipped.append(tuple(coords_arr[nearest_idx]))
        return flipped

    @staticmethod
    def generate_addition_probability_map(reference_instance, predictions, entity_predictions, mask_patch_coords, img_shape, patch_size: int = 16, alpha: float = 2.0, max_entity_overlap: float = 0.7, distance_penalty_weight: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Generate a 2D probability map for addition artifact candidates using perimeter patches
        
        Args:
            reference_instance: The reference instance
            predictions: GSAM model predictions  
            entity_predictions: GSAM model entity predictions
            mask_patch_coords: List of patch coordinates for the reference instance
            img_shape: Image shape (height, width, channels)
            patch_size: Size of patches (default 16)
            alpha: Distance extension parameter for perimeter ring (default 2.0)
            max_entity_overlap: Maximum acceptable entity overlap before penalty (default 0.7)
            distance_penalty_weight: Weight for distance penalty (default 0.1)
            
        Returns:
            Tuple of (probability_map, metadata_dict)
            - probability_map: 2D array with normalized probabilities
            - metadata_dict: Contains analysis results including sampled_patch and sampled_offset
        """
        H, W = img_shape[:2]
        
        patch_H = (H // patch_size) * patch_size
        patch_W = (W // patch_size) * patch_size
        patch_w = patch_W // patch_size
        patch_h = patch_H // patch_size
        
        # Initialize probability map
        probability_map = np.zeros((patch_h, patch_w), dtype=np.float32)
        
        # Get reference bbox and mask
        ref_bbox = reference_instance['pred_box'].cpu().numpy()
        ref_mask = reference_instance['pred_mask'].cpu().numpy()
        ref_entity_name = reference_instance['mapped_entity_name']
        
        # Convert reference mask to patch-aligned version for consistent calculations throughout
        # This ensures IoU calculations are performed at patch granularity, matching FLUX's patch-based approach
        ref_mask_patch = InstanceProcessor._align_mask_to_patches(ref_mask.astype(np.uint8), patch_size)

        # Get reference class from sampled instance
        reference_class_idx = reference_instance['pred_class'].item()
        
        # Step 1: Find best entity (vocab[0]) with highest overlap with reference_instance
        entity_instances = []
        for i, pred_instance in enumerate(entity_predictions):
            if ref_entity_name == pred_instance['entity_name']:
                entity_instances.append({
                    'idx': i,
                    'bbox': pred_instance['pred_box'].cpu().numpy(),
                    'mask': pred_instance['pred_mask'].cpu().numpy()
                })
        
        if not entity_instances:
            raise ValueError("No entity instances found")
        
        # Find entity with highest overlap with reference
        best_entity = None
        max_overlap = 0.9
        
        for entity in entity_instances:
            intersection = np.sum(ref_mask & entity['mask'])
            ref_area = np.sum(ref_mask)
            overlap = intersection / ref_area if ref_area > 0 else 0.0
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_entity = entity
        
        if best_entity is None:
            raise ValueError("No overlapping entity found")
        
        # Step 2: Get same class instances (excluding reference)
        reference_class_idx = reference_instance['pred_class'].item()
        same_class_instances = []
        
        for i, pred_instance in enumerate(predictions):
            if (pred_instance['pred_class'] == reference_class_idx and 
                not torch.equal(pred_instance['pred_box'], torch.from_numpy(ref_bbox).float())):
                same_class_instances.append({
                    'idx': i,
                    'bbox': pred_instance['pred_box'].cpu().numpy(),
                    'mask': pred_instance['pred_mask'].cpu().numpy()
                })
        
        # Step 3: Calculate reference center directly in patch coordinates
        ref_center_patch_y = int((ref_bbox[1] + ref_bbox[3]) / 2 // patch_size)
        ref_center_patch_x = int((ref_bbox[0] + ref_bbox[2]) / 2 // patch_size)
        
        # Step 4: Find perimeter patches using outermost patches and center distance
        reference_patch_set = set(mask_patch_coords)
        
        # Step 4-1: Find outermost patches of the reference segmentation
        outermost_patches = set()
        
        for ref_py, ref_px in mask_patch_coords:
            # Check if this patch is on the boundary (has at least one neighbor not in reference set)
            is_outermost = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_py = ref_py + dy
                    neighbor_px = ref_px + dx
                    
                    # If neighbor is within bounds and not in reference set, this is an outermost patch
                    if (0 <= neighbor_py < patch_h and 0 <= neighbor_px < patch_w and 
                        (neighbor_py, neighbor_px) not in reference_patch_set):
                        is_outermost = True
                        break
                if is_outermost:
                    break
            
            if is_outermost:
                outermost_patches.add((ref_py, ref_px))
        
        # Step 4-2: Calculate distance from outermost patches to center patch
        if not outermost_patches:
            # If no outermost patches found, use all reference patches
            outermost_patches = reference_patch_set
        
        # Step 4-3: Define perimeter patches using closest outermost patch approach
        perimeter_patches = set()
        
        for py in range(patch_h):
            for px in range(patch_w):
                # Skip if this patch is already a reference patch
                if (py, px) in reference_patch_set:
                    continue
                
                # Find closest outermost patch to this (py, px)
                min_distance_to_outermost = float('inf')
                closest_outermost_patch = None
                
                for outer_py, outer_px in outermost_patches:
                    distance_to_outermost = abs(py - outer_py) + abs(px - outer_px)
                    if distance_to_outermost < min_distance_to_outermost:
                        min_distance_to_outermost = distance_to_outermost
                        closest_outermost_patch = (outer_py, outer_px)
                
                # Calculate distance from closest outermost patch to center
                if closest_outermost_patch is not None:
                    closest_outer_py, closest_outer_px = closest_outermost_patch
                    outermost_to_center_distance = abs(closest_outer_py - ref_center_patch_y) + abs(closest_outer_px - ref_center_patch_x)
                    
                    # Calculate distance from current patch to center
                    patch_to_center_distance = abs(py - ref_center_patch_y) + abs(px - ref_center_patch_x)
                    
                    # Check if current patch is in the perimeter ring relative to its closest outermost patch
                    min_perimeter_distance = outermost_to_center_distance
                    max_perimeter_distance = outermost_to_center_distance + alpha
                    
                    if min_perimeter_distance < patch_to_center_distance <= max_perimeter_distance:
                        perimeter_patches.add((py, px))
        
        # Step 5: For each perimeter patch, calculate translation and probability
        total_candidates = len(perimeter_patches)
        valid_candidates = 0
        
        for perimeter_py, perimeter_px in perimeter_patches:
            # Step 5-1: Calculate translation vector from reference center to perimeter patch
            translation_y_patches = perimeter_py - ref_center_patch_y
            translation_x_patches = perimeter_px - ref_center_patch_x
            
            # Convert to pixel translation
            translation_y_pixels = translation_y_patches * patch_size
            translation_x_pixels = translation_x_patches * patch_size
            
            # Step 5-2: Generate candidate target mask by translating reference mask
            target_mask = np.zeros((H, W), dtype=np.uint8)
            
            # Find reference mask pixels
            ref_y_coords, ref_x_coords = np.where(ref_mask > 0)
            
            # Apply translation to coordinates
            target_y_coords = ref_y_coords + translation_y_pixels
            target_x_coords = ref_x_coords + translation_x_pixels
            
            # Filter coordinates that are within image bounds
            valid_mask = ((target_y_coords >= 0) & (target_y_coords < H) & 
                         (target_x_coords >= 0) & (target_x_coords < W))
            
            valid_target_y = target_y_coords[valid_mask]
            valid_target_x = target_x_coords[valid_mask]
            
            # Step 5-3: Check feasibility - skip if too few pixels remain
            feasibility_ratio = len(valid_target_y) / np.sum(ref_mask > 0)
            if feasibility_ratio < 0.5:  # Less than 50% of pixels are feasible
                continue
            
            # Set target mask pixels
            target_mask[valid_target_y, valid_target_x] = 1
            target_area = np.sum(target_mask > 0)
            
            if target_area == 0:
                continue
            
            # Step 5-4: Calculate three IoU-based scores
            
            # Convert entity mask to patch-aligned version for consistent calculations
            entity_mask_patch = InstanceProcessor._align_mask_to_patches(best_entity['mask'].astype(np.uint8), patch_size)
            
            # IoU with entity (positive contribution) - exclude reference mask area
            entity_mask_excluding_ref = entity_mask_patch & (~ref_mask_patch)
            entity_intersection = np.sum(target_mask & (entity_mask_excluding_ref > 0))
            entity_area = np.sum(entity_mask_excluding_ref > 0)
            entity_overlap = entity_intersection / entity_area if entity_area > 0 else 0.0
            
            if entity_overlap == 0.0:
                continue

            valid_candidates += 1
            
            # IoU with reference instance (negative contribution)
            ref_intersection = np.sum(target_mask & (ref_mask_patch > 0))
            ref_overlap = ref_intersection / target_area
            
            # IoU with same class instances (negative contribution)
            max_same_class_overlap = 0.0
            for same_inst in same_class_instances:
                same_inst_mask_patch = InstanceProcessor._align_mask_to_patches(same_inst['mask'].astype(np.uint8), patch_size)
                same_intersection = np.sum(target_mask & (same_inst_mask_patch > 0))
                same_overlap = same_intersection / target_area
                max_same_class_overlap = max(max_same_class_overlap, same_overlap)
            
            # Negative contributions (penalize overlap with reference and same class)
            
            # Step 5-7: Apply distance penalty based on moved distance
            moved_distance = abs(perimeter_py - ref_center_patch_y) + abs(perimeter_px - ref_center_patch_x)
            distance_penalty = 1 + (distance_penalty_weight * moved_distance)
            
            probability_score = (3 - ref_overlap - max_same_class_overlap - entity_overlap) / distance_penalty
            
            # Step 5-8: Update probability map at perimeter patch location
            probability_map[perimeter_py, perimeter_px] = probability_score
        
        # Step 6: Apply min-max normalization to non-zero components only
        non_zero_mask = probability_map > 0
        if np.any(non_zero_mask):
            non_zero_values = probability_map[non_zero_mask]
            min_prob = np.min(non_zero_values)
            max_prob = np.max(non_zero_values)
            
            if max_prob > min_prob:
                # Apply min-max normalization: (x - min) / (max - min)
                probability_map[non_zero_mask] = (non_zero_values - min_prob) / (max_prob - min_prob)
            else:
                # If all non-zero values are the same, set them to 1
                probability_map[non_zero_mask] = 1.0
        else:
            max_prob = 0
        
        # Sample a patch coordinate from the probability map
        sampled_patch = None
        sampled_offset = None
        if np.any(non_zero_mask):
            # Flatten probability map and get valid indices
            flat_prob_map = probability_map.flatten()
            valid_indices = np.where(flat_prob_map > 0)[0]
            valid_probabilities = flat_prob_map[valid_indices]
            
            # Select the index with maximum probability
            max_prob_idx = np.argmax(valid_probabilities)
            sampled_flat_idx = valid_indices[max_prob_idx]
            
            # Convert flat index back to 2D coordinates
            sampled_py, sampled_px = np.unravel_index(sampled_flat_idx, probability_map.shape)
            sampled_patch = (sampled_py, sampled_px)
            
            # Calculate offset from reference center to sampled patch
            offset_y_patches = sampled_py - ref_center_patch_y
            offset_x_patches = sampled_px - ref_center_patch_x
            offset_y_pixels = offset_y_patches * patch_size
            offset_x_pixels = offset_x_patches * patch_size
            sampled_offset = (offset_x_pixels, offset_y_pixels)
        
        # Step 7: Create metadata
        metadata = {
            'best_entity': best_entity,
            'same_class_count': len(same_class_instances),
            'total_candidates_tested': total_candidates,
            'valid_candidates': valid_candidates,
            'reference_center_patch': (ref_center_patch_y, ref_center_patch_x),
            'alpha': alpha,
            'max_entity_overlap': max_entity_overlap,
            'distance_penalty_weight': distance_penalty_weight,
            'num_outermost_patches': len(outermost_patches),
            'num_perimeter_patches': len(perimeter_patches),
            'probability_map_shape': probability_map.shape,
            'max_probability': max_prob if np.any(non_zero_mask) else 0,
            'min_probability': min_prob if np.any(non_zero_mask) else 0,
            'num_non_zero_patches': np.sum(non_zero_mask),
            'sampled_patch': sampled_patch,
            'sampled_offset': sampled_offset
        }
        
        return sampled_offset, probability_map, metadata

    @staticmethod
    def visualize_addition_probability_map(img_array: np.ndarray, probability_map: np.ndarray, 
                                         reference_instance, metadata: Dict,
                                         patch_size: int = 16, output_dir: str = None, 
                                         img_filename: str = None, colormap: str = 'hot',
                                         alpha: float = 0.6) -> None:
        """
        Visualize the addition probability map overlaid on the original image
        
        Args:
            img_array: Original image array
            probability_map: 2D probability map from generate_addition_probability_map
            reference_instance: The reference instance used to generate the map
            metadata: Metadata dictionary from generate_addition_probability_map
            patch_size: Size of patches (default 16)
            output_dir: Output directory for saving visualization
            img_filename: Image filename for output naming
            colormap: Matplotlib colormap for heatmap (default 'hot')
            alpha: Transparency of heatmap overlay (default 0.6)
        """
        H, W = img_array.shape[:2]
        patch_h, patch_w = probability_map.shape
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Original image with reference instance
        axes[0].imshow(img_array)
        
        # Overlay reference instance bounding box
        ref_bbox = reference_instance['pred_box'].cpu().numpy()
        rect = patches.Rectangle(
            (ref_bbox[0], ref_bbox[1]), 
            ref_bbox[2] - ref_bbox[0], 
            ref_bbox[3] - ref_bbox[1],
            linewidth=2, edgecolor='red', facecolor='none', label='Reference Instance'
        )
        axes[0].add_patch(rect)
        
        # Show reference center point
        ref_center_y, ref_center_x = metadata['reference_center_patch']
        ref_center_pixel_y = ref_center_y * patch_size + patch_size // 2
        ref_center_pixel_x = ref_center_x * patch_size + patch_size // 2
        axes[0].plot(ref_center_pixel_x, ref_center_pixel_y, 'r*', markersize=15, label='Reference Center')
        
        axes[0].set_title('Original Image with Reference Instance')
        axes[0].axis('off')
        axes[0].legend()
        
        # Panel 2: Probability heatmap only
        im1 = axes[1].imshow(probability_map, cmap=colormap, vmin=0, vmax=1)
        non_zero_count = metadata.get('num_non_zero_patches', 0)
        min_prob = metadata.get('min_probability', 0)
        max_prob = metadata.get('max_probability', 0)
        axes[1].set_title(f'Probability Heatmap (Min-Max Normalized)\nNon-zero patches: {non_zero_count}, Range: [{min_prob:.3f}, {max_prob:.3f}]')
        
        # Add patch grid lines
        for i in range(patch_h + 1):
            axes[1].axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(patch_w + 1):
            axes[1].axvline(x=j - 0.5, color='white', linewidth=0.5, alpha=0.3)
        
        # Mark reference center position
        axes[1].plot(ref_center_x, ref_center_y, 'w*', markersize=15, label='Reference Center')
        
        # Mark sampled patch if available
        if metadata.get('sampled_patch') is not None:
            sampled_py, sampled_px = metadata['sampled_patch']
            axes[1].plot(sampled_px, sampled_py, 'o', markersize=12, markerfacecolor='none', 
                        markeredgecolor='black', markeredgewidth=2, label='Sampled Patch')
        axes[1].legend()
        
        # Add colorbar for heatmap
        cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.8)
        cbar1.set_label('Probability', rotation=270, labelpad=20)
        
        # Panel 3: Overlay heatmap on original image
        axes[2].imshow(img_array)
        
        # Overlay reference patches
        ref_mask = reference_instance['pred_mask'].cpu().numpy()
        axes[2].imshow(ref_mask, alpha=0.3, cmap='Greens', vmin=0, vmax=1)
        
        # Resize probability map to match image dimensions
        probability_resized = np.zeros((H, W))
        for py in range(patch_h):
            for px in range(patch_w):
                y_start = py * patch_size
                y_end = min((py + 1) * patch_size, H)
                x_start = px * patch_size
                x_end = min((px + 1) * patch_size, W)
                probability_resized[y_start:y_end, x_start:x_end] = probability_map[py, px]
        
        # Create masked array to only show non-zero probabilities
        probability_masked = np.ma.masked_where(probability_resized == 0, probability_resized)
        
        # Overlay heatmap
        im2 = axes[2].imshow(probability_masked, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
        
        # Overlay reference instance bounding box
        rect2 = patches.Rectangle(
            (ref_bbox[0], ref_bbox[1]), 
            ref_bbox[2] - ref_bbox[0], 
            ref_bbox[3] - ref_bbox[1],
            linewidth=2, edgecolor='cyan', facecolor='none', label='Reference Instance'
        )
        axes[2].add_patch(rect2)
        
        # Show reference center point
        axes[2].plot(ref_center_pixel_x, ref_center_pixel_y, 'c*', markersize=15, label='Reference Center')
        
        # Mark sampled patch if available
        if metadata.get('sampled_patch') is not None:
            sampled_py, sampled_px = metadata['sampled_patch']
            sampled_pixel_y = sampled_py * patch_size + patch_size // 2
            sampled_pixel_x = sampled_px * patch_size + patch_size // 2
            axes[2].plot(sampled_pixel_x, sampled_pixel_y, 'ko', markersize=12, markerfacecolor='yellow', 
                        markeredgecolor='black', markeredgewidth=2, label='Sampled Patch')
        
        axes[2].set_title(f'Min-Max Normalized Probability Overlay\n({metadata["valid_candidates"]}/{metadata["total_candidates_tested"]} valid candidates)\nGreen: Reference, Hot: Probability, Yellow: Sampled Patch')
        axes[2].axis('off')
        axes[2].legend()
        
        # Add colorbar for overlay
        cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.8)
        cbar2.set_label('Probability', rotation=270, labelpad=20)
        
        # Add overall title with metadata
        sampled_info = ""
        if metadata.get('sampled_offset') is not None:
            offset_x, offset_y = metadata['sampled_offset']
            sampled_info = f", Sampled Offset: ({offset_x:.1f}, {offset_y:.1f})"
        
        fig.suptitle(f'Addition Probability Map Analysis (Min-Max Normalized)\n'
                    f'Alpha: {metadata["alpha"]}, Max Entity Overlap: {metadata["max_entity_overlap"]}, '
                    f'Distance Penalty: {metadata["distance_penalty_weight"]}{sampled_info}\n'
                    f'Valid Candidates: {metadata["valid_candidates"]}/{metadata["total_candidates_tested"]}, '
                    f'Non-zero: {metadata["num_non_zero_patches"]}', 
                    fontsize=11)
        
        plt.tight_layout()
        
        # Save visualization if output directory is provided
        if output_dir and img_filename:
            # Use output_dir directly (no additional subdirectory creation)
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "04_addition_probability_map.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Probability map visualization saved to: {output_path}")
        
        plt.close()