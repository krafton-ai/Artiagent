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
    def filter_instances_by_size(predictions: Dict, min_area_ratio: float = 0.01, max_area_ratio: float = 1.0) -> Tuple[any, torch.Tensor]: # type: ignore
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
    def sample_instance_by_score(predictions: Dict, min_area_ratio: float = 0.01, max_area_ratio: float = 1.0) -> Tuple[Optional[any], Optional[int]]: # type: ignore
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
    def create_artifact_patches(artifact_type: str, prediction: Dict, predictions: Dict, entity_predictions: Dict, img_array, patch_size: int = 16, distortion_kernel: str = 'none', output_dir: str = None, img_filename: str = None) -> Tuple[List[int], List[int], Optional[Dict]]:
        """
        Create artifact patches for different artifact types

        Target-reference mapping by artifact type:
        - Addition: target = reference mask patches shifted by the sampled offset; reference = original mask patches (one-to-one alignment by order).
        - Removal: target = mask patches of the subentity (area to remove); reference = nearby non-foreground, non-conflicting patches just outside the mask (within 1-Manhattan ring).
        - Distortion: target = mask patches; reference = kernel-remapped coordinates of the same mask patches (none → [], shuffle → permutation, jitter → Gaussian jitter within foreground, swirl → swirl around centroid, voronoi → nearest-seed remap, coarse → partition-based shuffle, strip → circular strip shifts).

        Returns:
            Tuple containing:
            - target_patches: List of target patch coordinates
            - reference_patches: List of reference patch coordinates  
            - metadata: Optional dictionary with additional information (e.g., fusion metadata)
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
            # InstanceProcessor.visualize_addition_probability_map(img_array, prob_map, prediction, metadata, patch_size=patch_size, output_dir=output_dir, img_filename=img_filename)

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
                        if abs(dy) + abs(dx) <= 2:  # Hamilton distance <= 3
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
                if prediction['entity'] == entity_pred_instance['entity']:
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
            
            # Pre-compute set of patches that contain ANY prediction instance pixels (foreground patches)
            foreground_patches = set()
            for entity_pred_instance in entity_predictions:
                if prediction['entity'] == entity_pred_instance['entity']:
                    entity_mask = entity_pred_instance['pred_mask'].cpu().numpy()
                    entity_patch_indices = mask_to_patch_indices(entity_mask, patch_size=patch_size, txt_len=512)
                    entity_patch_coords = patch_indices_to_coords(entity_patch_indices, patch_w, txt_len=512)
                    entity_patch_coords = [tuple(coord) for coord in entity_patch_coords]
                    foreground_patches.update(entity_patch_coords)
            
            # Apply distortion kernel based on specified type
            if distortion_kernel == 'none':
                reference_patches = []
            elif distortion_kernel == 'shuffle':
                reference_patches = mask_patch_coords.copy()
                random.shuffle(reference_patches)
            elif distortion_kernel == 'jitter':
                # Apply Gaussian jitter kernel with foreground patch constraint
                reference_patches = InstanceProcessor.gaussian_jitter_kernel(
                    mask_patch_coords, sigma=1.0, patch_h=patch_h, patch_w=patch_w,
                    foreground_patches=foreground_patches
                )
            elif distortion_kernel == 'swirl':
                # Apply swirl kernel
                reference_patches = InstanceProcessor.swirl_kernel(
                    mask_patch_coords, strength=0.3, patch_h=patch_h, patch_w=patch_w
                )
            elif distortion_kernel == 'voronoi':
                # Apply Voronoi seed kernel
                reference_patches = InstanceProcessor.voronoi_seed_kernel(
                    mask_patch_coords, seed_fraction=0.15, patch_h=patch_h, patch_w=patch_w
                )
            elif distortion_kernel == 'coarse':
                # Apply coarse shuffling kernel
                reference_patches = InstanceProcessor.coarse_shuffling_kernel(
                    mask_patch_coords, num_partitions=5, patch_h=patch_h, patch_w=patch_w   
                )
            elif distortion_kernel == 'strip':
                # Apply stripped shifting kernel
                reference_patches = InstanceProcessor.stripped_shifting_kernel(
                    mask_patch_coords, num_strips=6, patch_h=patch_h, patch_w=patch_w
                )
            else:
                raise ValueError(f"Unknown distortion kernel: {distortion_kernel}")
            
            print(f"Applied {distortion_kernel} distortion kernel: {len(mask_patch_coords)} -> {len(reference_patches)} patches")

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        return target_patches, reference_patches
    
    @staticmethod
    def create_fusion_artifact_patches(entity_prediction, entity_predictions, predictions, img_array, patch_size: int = 16, output_dir: str = None, img_filename: str = None) -> Tuple[List[int], List[int], Optional[Dict]]:
        """
        Create fusion artifact patches for a detected entity-subentity combination.
        
        This function implements fusion artifact logic:
        1. Find overlapping entity instances
        2. Filter by containment ratio (< 0.5)
        3. Select entity with highest overlap
        4. Create target patches (overlapping region + 1-Manhattan ring, foreground-only)
        5. Create reference patch pool (2–4 Manhattan distance, foreground-only; exclude targets)
        6. Target-reference mapping:
           - A-only targets → nearest patches from B-only reference pool (fallback to full pool)
           - B-only targets → nearest patches from A-only reference pool (fallback to full pool)
           - Overlap targets → nearest patches from the full reference pool
        7. Symmetric augmentation: add reversed (reference→target) mappings without introducing duplicate targets.
        """
        H, W = img_array.shape[:2]
        patch_H = (H // patch_size) * patch_size
        patch_W = (W // patch_size) * patch_size
        patch_w = patch_W // patch_size
        patch_h = patch_H // patch_size

        # Get entity A (current entity) information
        entity_A = entity_prediction['entity']
        mask_A = entity_prediction['pred_mask'].cpu().numpy()
        
        mask_A_patch_coords = mask_to_patch_coords(mask_A, patch_size=patch_size)
        mask_A_patch_set = set(mask_A_patch_coords)

        # Step 1: Find overlapping instances for the same entity
        overlapping_candidates = []
        for i, other_prediction in enumerate(entity_predictions):
            # Skip if same instance
            if torch.equal(entity_prediction['pred_box'], other_prediction['pred_box']):
                continue
            
            mask_B = other_prediction['pred_mask'].cpu().numpy()
            
            # Convert mask B to patch coordinates
            mask_B_patch_coords = mask_to_patch_coords(mask_B, patch_size=patch_size)
            mask_B_patch_set = set(mask_B_patch_coords)
            
            # Calculate intersection and areas using patch coordinate sets
            intersection_patches = mask_A_patch_set & mask_B_patch_set
            intersection = len(intersection_patches)
            area_A = len(mask_A_patch_set)
            area_B = len(mask_B_patch_set)
            
            if area_A == 0 or area_B == 0:
                continue
            
            # Containment ratio (how much each instance contains the other)
            containment_A_in_B = intersection / area_A
            containment_B_in_A = intersection / area_B
            
            # Filter out instances that contain one another (containment ratio >= 0.5)
            if containment_A_in_B >= 0.7 or containment_B_in_A >= 0.7:
                continue
            
            
            if intersection > 5:
                overlapping_candidates.append({
                    'index': i,
                    'prediction': other_prediction,
                    'mask_B_patch_set': mask_B_patch_set,
                    'intersection': intersection
                })

        if not overlapping_candidates:
            raise ValueError("No valid overlapping entity found for fusion artifact")

        # Step 2: Select entity with highest overlap
        best_candidate = max(overlapping_candidates, key=lambda x: x['intersection'])
        entity_B_prediction = best_candidate['prediction']
        mask_B_patch_set = best_candidate['mask_B_patch_set']
        mask_B_patch_coords = list(mask_B_patch_set)

        # Step 3: Create overlapping region using patch coordinate intersection
        overlap_patch_set = mask_A_patch_set & mask_B_patch_set
        overlap_patch_coords = list(overlap_patch_set)


        A_set = mask_A_patch_set
        B_set = mask_B_patch_set
        overlap_set = overlap_patch_set
        foreground_set = A_set | B_set

        # Non-overlap pools
        A_only = A_set - overlap_set
        B_only = B_set - overlap_set
        non_overlap_foreground = foreground_set - overlap_set

        if not overlap_set:
            return [], []

        band_R = 2
        k_seeds = 5
        rng = random.Random(2025)

        def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def _farthest_point_sampling(points: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
            if not points or k <= 0:
                return []
            pts = list(points)
            if k >= len(pts):
                return pts
            cy = sum(p[0] for p in pts) / len(pts)
            cx = sum(p[1] for p in pts) / len(pts)
            start = min(pts, key=lambda p: (p[0] - cy)**2 + (p[1] - cx)**2)
            seeds = [start]
            if k <= 1:
                return seeds
            min_distances = {p: _manhattan(p, start) for p in pts}
            for _ in range(1, k):
                nxt = max(pts, key=lambda p: min_distances[p])
                seeds.append(nxt)
                for p in pts:
                    d = _manhattan(p, nxt)
                    if d < min_distances[p]:
                        min_distances[p] = d
            return seeds

        # Build target region: band around overlap_set with Manhattan distance ≤ band_R
        band_targets = set()
        for oy, ox in overlap_set:
            for dy in range(-band_R, band_R + 1):
                for dx in range(-band_R, band_R + 1):
                    if _manhattan((0, 0), (dy, dx)) <= band_R:
                        ny, nx = oy + dy, ox + dx
                        if 0 <= ny < patch_h and 0 <= nx < patch_w and (ny, nx) in foreground_set:
                            band_targets.add((ny, nx))

        if not band_targets:
            return [], []

        # Seeds chosen from the overlap (centers) to partition the band
        seeds = _farthest_point_sampling(list(band_targets), k_seeds)
        if not seeds:
            return [], []

        # Determine which entity each seed is closer to
        def _get_seed_entity(seed: Tuple[int, int]) -> str:
            if not A_only and not B_only:
                return 'neutral'
            
            dist_to_A = min([_manhattan(seed, a) for a in A_only]) if A_only else float('inf')
            dist_to_B = min([_manhattan(seed, b) for b in B_only]) if B_only else float('inf')
            
            if dist_to_A < dist_to_B:
                return 'A'
            elif dist_to_B < dist_to_A:
                return 'B'
            else:
                return 'neutral'

        seed_entities = {seed: _get_seed_entity(seed) for seed in seeds}

        # Voronoi partition of the band
        regions = {seed: [] for seed in seeds}
        for patch in band_targets:
            nearest_seed = min(seeds, key=lambda s: _manhattan(patch, s))
            regions[nearest_seed].append(patch)

        # Candidate offsets in increasing L1 radius (1..4)
        candidate_offsets = []
        for l1_dist in range(1, 5):
            for dy in range(-l1_dist, l1_dist + 1):
                for dx in range(-l1_dist, l1_dist + 1):
                    if _manhattan((0, 0), (dy, dx)) == l1_dist:
                        candidate_offsets.append((dy, dx))

        target_coords_final: List[Tuple[int, int]] = []
        reference_coords_final: List[Tuple[int, int]] = []

        # Helper: nearest in non-overlap foreground (exclude overlap as reference)
        def _nearest_non_overlap(p: Tuple[int, int]) -> Tuple[int, int]:
            if not non_overlap_foreground:
                return p  # last resort
            return min(non_overlap_foreground, key=lambda q: _manhattan(p, q))

        # Helper: nearest in opposite entity
        def _nearest_opposite_entity(p: Tuple[int, int], seed_entity: str) -> Tuple[int, int]:
            if seed_entity == 'A' and B_only:
                return min(B_only, key=lambda q: _manhattan(p, q))
            elif seed_entity == 'B' and A_only:
                return min(A_only, key=lambda q: _manhattan(p, q))
            else:
                # Fallback to any non-overlap
                return _nearest_non_overlap(p)

        for seed in seeds:
            region_patches = regions[seed]
            if not region_patches:
                continue

            seed_entity = seed_entities[seed]
            
            # Determine target entity for references based on seed entity
            if seed_entity == 'A':
                target_entity_set = B_only
            elif seed_entity == 'B':
                target_entity_set = A_only
            else:
                # Neutral case - use either entity
                target_entity_set = non_overlap_foreground

            # --- Primary attempt: a single offset that maps the entire region
            #     into the opposite entity only AND doesn't overlap with target patches
            valid_offset = None
            for dy, dx in candidate_offsets:
                shifted = [(py + dy, px + dx) for (py, px) in region_patches]
                # bounds
                if not all(0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted):
                    continue
                shifted_set = set(shifted)
                # Check that shifted patches are in target entity AND don't overlap with any target patches
                if shifted_set.issubset(target_entity_set) and not shifted_set.intersection(band_targets):
                    valid_offset = (dy, dx)
                    break

            if valid_offset is not None:
                dy, dx = valid_offset
                for py, px in region_patches:
                    target_coords_final.append((py, px))
                    reference_coords_final.append((py + dy, px + dx))
                continue

            # --- Fallback: choose the offset that maps the MAX number of patches
            #     into the opposite entity without overlapping target patches
            best_offset = None
            best_mapped_count = -1
            best_mapped_mask = None  # list[bool] parallel to region_patches

            for dy, dx in candidate_offsets:
                shifted = [(py + dy, px + dx) for (py, px) in region_patches]
                # bounds mask
                in_bounds = [0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted]
                # opposite entity mask AND not in target patches
                opposite_entity_mask = [
                    in_bounds[i] and (shifted[i] in target_entity_set) and (shifted[i] not in band_targets)
                    for i in range(len(shifted))
                ]
                mapped_count = sum(opposite_entity_mask)
                if mapped_count > best_mapped_count:
                    best_mapped_count = mapped_count
                    best_offset = (dy, dx)
                    best_mapped_mask = opposite_entity_mask

            if best_offset is not None and best_mapped_count > 0:
                dy, dx = best_offset
                for i, (py, px) in enumerate(region_patches):
                    target_coords_final.append((py, px))
                    if best_mapped_mask[i]:  # pyright: ignore[reportOptionalSubscript]
                        # Use offset-mapped opposite entity reference
                        reference_coords_final.append((py + dy, px + dx))
                    else:
                        # Fill with nearest opposite entity reference
                        nearest = _nearest_opposite_entity((py, px), seed_entity)
                        # Make sure the nearest reference isn't in the target set
                        attempts = 0
                        candidates = list(target_entity_set - band_targets)
                        while nearest in band_targets and attempts < 5 and candidates:
                            # Try to find another reference not in target set
                            nearest = min(candidates, key=lambda q: _manhattan((py, px), q))
                            candidates.remove(nearest)
                            attempts += 1
                        reference_coords_final.append(nearest)
            else:
                # Nothing could be offset into opposite entity → all via nearest opposite entity
                for py, px in region_patches:
                    target_coords_final.append((py, px))
                    nearest = _nearest_opposite_entity((py, px), seed_entity)
                    # Make sure the nearest reference isn't in the target set
                    attempts = 0
                    candidates = list(target_entity_set - band_targets)
                    while nearest in band_targets and attempts < 5 and candidates:
                        # Try to find another reference not in target set
                        nearest = min(candidates, key=lambda q: _manhattan((py, px), q))
                        candidates.remove(nearest)
                        attempts += 1
                    reference_coords_final.append(nearest)

        return target_coords_final, reference_coords_final, entity_B_prediction['entity']
        # return target_coords_final, reference_coords_final, entity_B_prediction['entity'] # pyright: ignore[reportReturnType]
    
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
            target_patch_indices = [patch_coor_to_ind(tarPx, tar_py, patch_w, txt_len) for tar_py, tarPx in target_patches]
            return target_patch_indices, reference_patch_indices
        
        elif artifact_type == 'fusion':
            reference_patch_indices = [patch_coor_to_ind(ref_px, ref_py, patch_w, txt_len) for ref_py, ref_px in reference_patches]
            target_patch_indices = [patch_coor_to_ind(tarPx, tar_py, patch_w, txt_len) for tar_py, tarPx in target_patches]
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
            
    # for debug
    def visualize_target_mask(img_array: np.ndarray, masks_data: Dict,
                            img_filename: str, output_dir: str, patch_size: int = 16):
        viz_output_dir = output_dir
        os.makedirs(viz_output_dir, exist_ok=True)
        
        for artifact_type, masks in masks_data.items():
            if 'error' in masks:
                continue
                
            reference_mask = masks.get('reference_mask')
            target_mask = masks.get('target_mask')

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_array)
            ax.imshow(target_mask, alpha=0.5, cmap='Blues')

            # set minor ticks at patch boundaries
            height, width = img_array.shape[:2]
            ax.set_xticks(np.arange(0, width, patch_size), minor=True)
            ax.set_yticks(np.arange(0, height, patch_size), minor=True)

            # # draw grid
            ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

            # turn off all ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(viz_output_dir, "00_target_mask.png"), 
                            dpi=150, bbox_inches='tight')
            plt.close(fig)

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
        foreground_patches: Optional[set] = None,
    ) -> List[Tuple[int, int]]:
        """
        Apply i.i.d. Gaussian jitter to each patch coordinate.

        Args:
            mask_patch_coords: List of (py, px) integers to be distorted.
            sigma: Standard deviation of the normal noise added to each axis, in patch units.
            patch_h, patch_w: Optional explicit patch‑grid height/width.  If not given they
                are inferred from the maximum coordinates in `mask_patch_coords`.
            foreground_patches: Optional set of valid foreground patch coordinates. If provided,
                jittered coordinates are constrained to stay within this set.

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
            # Try to find a valid jittered coordinate within foreground patches
            attempts = 0
            max_attempts = 10  # Limit attempts to avoid infinite loops
            
            while attempts < max_attempts:
                n_py = int(round(py + np.random.normal(0, sigma)))
                n_px = int(round(px + np.random.normal(0, sigma)))
                n_py = max(0, min(n_py, patch_h - 1))
                n_px = max(0, min(n_px, patch_w - 1))
                
                # If foreground_patches is provided, check if jittered coordinate is valid
                if foreground_patches is None or (n_py, n_px) in foreground_patches:
                    jittered.append((n_py, n_px))
                    break
                    
                attempts += 1
            
            # If no valid jittered coordinate found within attempts, fall back to original
            if attempts >= max_attempts:
                if foreground_patches is None or (py, px) in foreground_patches:
                    jittered.append((py, px))
                else:
                    # Find nearest foreground patch as fallback
                    min_dist = float('inf')
                    nearest_patch = (py, px)
                    for fg_py, fg_px in foreground_patches:
                        dist = abs(py - fg_py) + abs(px - fg_px)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_patch = (fg_py, fg_px)
                    jittered.append(nearest_patch)
                    
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
        rot = 1 if random.random() < 0.5 else -1
        for py, px in mask_patch_coords:
            dy, dx = py - cy, px - cx
            r = np.hypot(dy, dx)
            theta = np.arctan2(dy, dx) + rot * strength / max(0.1, (r / max_r))
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
    def coarse_shuffling_kernel(
        mask_patch_coords: List[Tuple[int, int]],
        num_partitions: int = 5,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        rng: Optional[random.Random] = None
    ) -> List[Tuple[int, int]]:
        """
        Coarsely partition region and shuffle within regions. 

        Args:
            mask_patch_coords: Coordinates to remap.
            num_partitions: Number of partitions to generate.
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

        coords_arr = np.array(mask_patch_coords)
        region_set = set(mask_patch_coords)
        n_coords = len(mask_patch_coords)
        num_partitions = min(num_partitions, n_coords)

        # 1. Randomly select seeds
        seeds = rng.sample(mask_patch_coords, num_partitions)
        seeds_arr = np.array(seeds)

        # 2. Assign each coordinate to the nearest seed (partition)
        dists = ((coords_arr[:, None, :] - seeds_arr[None, :, :]) ** 2).sum(-1)
        partition_indices = dists.argmin(axis=1)  # For each coord, index of nearest seed

        # Build partition lists
        partitions = [[] for _ in range(num_partitions)]
        for idx, coord in enumerate(mask_patch_coords):
            part_idx = partition_indices[idx]
            partitions[part_idx].append(coord)

        # 3. Create a random mapping between partitions (no self-mapping)
        partition_mapping = list(range(num_partitions))
        rng.shuffle(partition_mapping)
        # Ensure no partition maps to itself
        for i in range(num_partitions):
            if partition_mapping[i] == i:
                # Find another partition to swap with
                for j in range(num_partitions):
                    if j != i and partition_mapping[j] != i:
                        partition_mapping[i], partition_mapping[j] = partition_mapping[j], partition_mapping[i]
                        break

        # 4. For each coordinate, reference from the mapped partition
        mapped_coords = []
        for idx, coord in enumerate(mask_patch_coords):
            current_partition = partition_indices[idx]
            target_partition = partition_mapping[current_partition]
            if partitions[target_partition]:
                candidate = rng.choice(partitions[target_partition])
                if candidate in region_set:
                    mapped_coords.append(candidate)
                else:
                    # Fallback to nearest coordinate in region
                    dists = np.sum((coords_arr - np.array(candidate)) ** 2, axis=1)
                    nearest_idx = np.argmin(dists)
                    mapped_coords.append(tuple(coords_arr[nearest_idx]))
            else:
                mapped_coords.append(coord)
        return mapped_coords

    @staticmethod
    def stripped_shifting_kernel(
        mask_patch_coords: List[Tuple[int, int]],
        num_strips: int = 4,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        patch_size: Optional[int] = 16,
        direction: Optional[str] = None
    ) -> List[Tuple[int, int]]:
        """
        Vertically / horizontally cut into strips and shift with different strengths.

        Args:
            mask_patch_coords: Coordinates to remap.
            num_strips: Number of strips to generate.
            patch_h, patch_w: Explicit grid size for clamping (inferred if None).
            direction: Optional direction to determine which way to generate strips.

        Returns:
            List where each entry is the coordinate of the seed that the corresponding
            input patch copies.
        """
        if not mask_patch_coords:
            return []

        if patch_h is None or patch_w is None:
            max_py = max(py for py, _ in mask_patch_coords)
            max_px = max(px for _, px in mask_patch_coords)
            patch_h = max_py + 1
            patch_w = max_px + 1

        if direction is None:
            coords_arr = np.array(mask_patch_coords)
            x_min, y_min = coords_arr.min(axis=0)
            x_max, y_max = coords_arr.max(axis=0)
            width = x_max - x_min + 1
            height = y_max - y_min + 1

            aspect_ratio = width / height

            if aspect_ratio < 1:    # if horizontal region, strip vertically
                direction = 'vertical'  
            else:                   # if vertical region, strip horizontally
                direction = 'horizontal'
        # Create region set for fast lookup
        region_set = set(mask_patch_coords)
        coords_arr = np.array(mask_patch_coords)

        # Determine strip boundaries
        if direction == 'vertical':
            # Vertical strips: divide by x-coordinate
            x_coords = [px for _, px in mask_patch_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            strip_width = (x_max - x_min + 1) // num_strips
            
            # Create strip boundaries
            strip_boundaries = []
            for i in range(num_strips):
                start_x = x_min + i * strip_width
                end_x = x_min + (i + 1) * strip_width if i < num_strips - 1 else x_max + 1
                strip_boundaries.append((start_x, end_x))
        else:  # horizontal
            # Horizontal strips: divide by y-coordinate
            y_coords = [py for py, _ in mask_patch_coords]
            y_min, y_max = min(y_coords), max(y_coords)
            strip_height = (y_max - y_min + 1) // num_strips
            
            # Create strip boundaries
            strip_boundaries = []
            for i in range(num_strips):
                start_y = y_min + i * strip_height
                end_y = y_min + (i + 1) * strip_height if i < num_strips - 1 else y_max + 1
                strip_boundaries.append((start_y, end_y))

        # Assign each coordinate to a strip
        strip_assignments = []
        for py, px in mask_patch_coords:
            strip_idx = -1
            for i, (start, end) in enumerate(strip_boundaries):
                if direction == 'vertical':
                    if start <= px < end:
                        strip_idx = i
                        break
                else:  # horizontal
                    if start <= py < end:
                        strip_idx = i
                        break
            strip_assignments.append(strip_idx)

        # Generate shift strengths for each strip (different for each strip)
        shift_strengths = []
        for i in range(num_strips):
            direction = 1 if i % 2 == 0 else -1
            # Use different shift strengths: 1, 2, 3, 4 pixels etc.
            shift_strength = direction * (i + 1) * 10  # 10, 20, 30, 40 pixels
            shift_strengths.append(shift_strength)
        random.shuffle(shift_strengths)

        # Apply shifts to each coordinate
        shifted_coords = []
        strip_coords = [[] for _ in range(num_strips)]
        strip_indices = [[] for _ in range(num_strips)]
        
        for i, (py, px) in enumerate(mask_patch_coords):
            strip_idx = strip_assignments[i]
            if strip_idx >= 0:
                strip_coords[strip_idx].append((py, px))
                strip_indices[strip_idx].append(i)
        
        # Apply circular shift within each strip
        for strip_idx in range(num_strips):
            if not strip_coords[strip_idx]:
                continue
                
            strip_patches = strip_coords[strip_idx]
            strip_patch_indices = strip_indices[strip_idx]
            shift_strength = shift_strengths[strip_idx]
            
            # Sort patches within the strip for consistent ordering
            if direction == 'vertical':
                # Sort by y-coordinate (vertical position) for vertical strips
                sorted_indices = sorted(range(len(strip_patches)), 
                                    key=lambda idx: strip_patches[idx][0])
            else:  # horizontal
                # Sort by x-coordinate (horizontal position) for horizontal strips
                sorted_indices = sorted(range(len(strip_patches)), 
                                    key=lambda idx: strip_patches[idx][1])
            
            # Apply circular shift within the strip
            num_patches_in_strip = len(strip_patches)
            if num_patches_in_strip > 0:
                # Calculate how many positions to shift (modulo for circular effect)
                shift_positions = shift_strength // patch_size  # Convert pixel shift to patch shift
                shift_positions = shift_positions % num_patches_in_strip  # Ensure circular
                
                # Create circular mapping
                for i, original_idx in enumerate(sorted_indices):
                    # Calculate new position with circular shift
                    new_idx = (i + shift_positions) % num_patches_in_strip
                    shifted_patch = strip_patches[sorted_indices[new_idx]]
                    
                    # Find the original index in the full list
                    original_full_idx = strip_patch_indices[original_idx]
                    
                    # Ensure we have enough space in shifted_coords
                    while len(shifted_coords) <= original_full_idx:
                        shifted_coords.append(None)
                    
                    shifted_coords[original_full_idx] = shifted_patch
        
        # Fill in any missing coordinates (shouldn't happen, but safety check)
        for i, (py, px) in enumerate(mask_patch_coords):
            if i >= len(shifted_coords) or shifted_coords[i] is None:
                shifted_coords.append((py, px))

        return shifted_coords

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
        ref_entity = reference_instance['entity']
        
        # Convert reference mask to patch-aligned version for consistent calculations throughout
        # This ensures IoU calculations are performed at patch granularity, matching FLUX's patch-based approach
        ref_mask_patch = InstanceProcessor._align_mask_to_patches(ref_mask.astype(np.uint8), patch_size)

        # Get reference class from sampled instance
        reference_class_idx = reference_instance['pred_class'].item()
        
        # Step 1: Find best entity (vocab[0]) with highest overlap with reference_instance
        entity_instances = []
        for i, pred_instance in enumerate(entity_predictions):
            if ref_entity == pred_instance['entity']:
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

######### FUSION helper functions #########

    @staticmethod
    def build_bands(A_set, B_set, overlap_set, R=3):
        union_set = A_set | B_set
        band = set()
        for p in union_set:
            py, px = p
            for o in overlap_set:
                oy, ox = o
                if abs(py - oy) + abs(px - ox) <= R:
                    band.add(p)
                    break
        bandA = band & A_set
        bandB = band & B_set
        return bandA, bandB
    
    @staticmethod
    def _partition_band_by_seeds(band_list, k):
        if not band_list:
            return []
        
        k = max(1, min(k, len(band_list)))
        seeds_indices = random.sample(range(len(band_list)), k)
        seeds = [band_list[i] for i in seeds_indices]
        
        parts = [[] for _ in range(k)]
        for i, coord in enumerate(band_list):
            py, px = coord
            min_dist_sq = float('inf')
            best_seed_idx = 0
            for seed_idx, seed_coord in enumerate(seeds):
                sy, sx = seed_coord
                dist_sq = abs(py - sy) + abs(px - sx)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_seed_idx = seed_idx
            parts[best_seed_idx].append(i)
        
        return parts
    
    @staticmethod
    def _deranged_partition_map(kA, kB):
        """
        Build A->B partition mapping with oversampling and randomization:
        - If kA < kB: oversample A so every B is assigned; each A gets ~ceil(kB/kA) Bs.
        - If kA > kB: oversample B so every A gets a B; some Bs are reused.
        - If kA == kB: 1-to-1 random mapping.
        Returns: dict {a_partition_id: [list_of_b_partition_ids]}
        """
        if kA <= 0 or kB <= 0:
            return {}

        # Randomized orders to avoid structured bias
        A_ids = list(range(kA))
        B_ids = list(range(kB))
        random.shuffle(A_ids)
        random.shuffle(B_ids)

        # Prepare mapping: each A id maps to a (possibly multi) list of B ids
        map_dict = {a: [] for a in range(kA)}

        # Number of assignments to generate
        # - If kA < kB: generate kB assignments so all B partitions are covered
        # - If kA > kB: generate kA assignments so all A partitions get at least one B
        # - If equal: generate kA (== kB) assignments for a 1-1 mapping
        L = max(kA, kB)

        for i in range(L):
            a = A_ids[i % kA]
            b = B_ids[i % kB]
            map_dict[a].append(b)

        return map_dict
    
    @staticmethod
    def _pair_within_partitions(bandA_list, bandB_list, partsA, partsB, mapAtoB, cap_pairs=128, intra_pick='nearest'):
        targets = []
        refs = []
        used_B_indices = set()
        
        for a_part_id in range(len(partsA)):
            if len(targets) >= cap_pairs:
                break
                
            a_indices = partsA[a_part_id][:]
            
            for a_idx in a_indices:
                if len(targets) >= cap_pairs:
                    break
                
                # Get mapped B partitions
                mapped_b_parts = mapAtoB.get(a_part_id, [])
                
                best_b_idx = None
                if intra_pick == 'nearest':
                    a_coord = bandA_list[a_idx]
                    ay, ax = a_coord
                    min_dist = float('inf')
                    
                    for b_part_id in mapped_b_parts:
                        for b_idx in partsB[b_part_id]:
                            if b_idx in used_B_indices:
                                continue
                            b_coord = bandB_list[b_idx]
                            by, bx = b_coord
                            dist = abs(ay - by) + abs(ax - bx)
                            if dist < min_dist:
                                min_dist = dist
                                best_b_idx = b_idx
                
                # Fallback to any remaining B index
                if best_b_idx is None:
                    for b_part_id in mapped_b_parts:
                        for b_idx in partsB[b_part_id]:
                            if b_idx not in used_B_indices:
                                best_b_idx = b_idx
                                break
                        if best_b_idx is not None:
                            break
                
                # Last resort fallback
                if best_b_idx is None:
                    for b_idx in range(len(bandB_list)):
                        if b_idx not in used_B_indices:
                            best_b_idx = b_idx
                            break
                
                if best_b_idx is not None:
                    targets.append(bandA_list[a_idx])
                    refs.append(bandB_list[best_b_idx])
                    used_B_indices.add(best_b_idx)
        
        return targets, refs
    
########## END OF FUSION helper functions #########