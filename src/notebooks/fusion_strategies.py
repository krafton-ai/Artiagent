from typing import List, Tuple, Dict
import random

def strategy_1(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_H = H
    patch_W = W
    patch_w = patch_W // patch_size
    patch_h = patch_H // patch_size
    mask_A_patch_coords = [tuple(coord) for coord in data['mask_A_patch_coords']]
    mask_B_patch_coords = [tuple(coord) for coord in data['mask_B_patch_coords']]
    overlap_patch_coords = [tuple(coord) for coord in data['overlap_patch_coords']]
    mask_A_patch_set = set(mask_A_patch_coords)
    mask_B_patch_set = set(mask_B_patch_coords)
    overlap_patch_set = set(overlap_patch_coords)

    target_patch_set = set(overlap_patch_coords)
    for oy, ox in overlap_patch_coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if abs(dy) + abs(dx) <= 1:  # Manhattan distance <= 1
                    ny, nx = oy + dy, ox + dx
                    if 0 <= ny < patch_h and 0 <= nx < patch_w:
                        target_patch_set.add((ny, nx))
    # Ensure all target patches are foreground patches (part of either entity A or B)
    foreground_patch_set = mask_A_patch_set | mask_B_patch_set
    target_patch_set = target_patch_set & foreground_patch_set
    target_patch_coords = list(target_patch_set)

    # Step 5: Partition target region into three parts
    A_only_coords = [coord for coord in target_patch_coords if coord in mask_A_patch_set and coord not in overlap_patch_set]
    B_only_coords = [coord for coord in target_patch_coords if coord in mask_B_patch_set and coord not in overlap_patch_set]
    overlap_target_coords = [coord for coord in target_patch_coords if coord in overlap_patch_set]

    # Step 6: Create reference patch pool (2-3 Manhattan distance from target patches)
    reference_patch_set = set()
    for ty, tx in target_patch_coords:
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                manhattan_dist = abs(dy) + abs(dx)
                if 2 <= manhattan_dist <= 4:  # Manhattan distance between 2-3
                    ny, nx = ty + dy, tx + dx
                    if 0 <= ny < patch_h and 0 <= nx < patch_w:
                        reference_patch_set.add((ny, nx))

    # Ensure reference patches are foreground patches
    reference_patch_set = reference_patch_set & foreground_patch_set
    # Remove target patches from reference pool
    reference_patch_set = reference_patch_set - target_patch_set

    # Create region-specific reference pools
    A_only_ref_coords = [coord for coord in reference_patch_set if coord in mask_A_patch_set and coord not in mask_B_patch_set]
    B_only_ref_coords = [coord for coord in reference_patch_set if coord in mask_B_patch_set and coord not in mask_A_patch_set]
    full_ref_coords = list(reference_patch_set)

    # Step 7: Map target patches to reference patches according to sampling strategy
    target_coords_final = []
    reference_coords_final = []

    # For A entity only target region → sample from B entity only reference patch pool
    for coord in A_only_coords:
        target_coords_final.append(coord)
        if B_only_ref_coords:
            # Find closest patch in B-only reference pool
            best_ref = min(B_only_ref_coords, key=lambda ref: abs(coord[0] - ref[0]) + abs(coord[1] - ref[1]))
            reference_coords_final.append(best_ref)
        elif full_ref_coords:
            # Fallback to full reference pool
            best_ref = min(full_ref_coords, key=lambda ref: abs(coord[0] - ref[0]) + abs(coord[1] - ref[1]))
            reference_coords_final.append(best_ref)

    # For B entity only target region → sample from A entity only reference patch pool
    for coord in B_only_coords:
        target_coords_final.append(coord)
        if A_only_ref_coords:
            # Find closest patch in A-only reference pool
            best_ref = min(A_only_ref_coords, key=lambda ref: abs(coord[0] - ref[0]) + abs(coord[1] - ref[1]))
            reference_coords_final.append(best_ref)
        elif full_ref_coords:
            # Fallback to full reference pool
            best_ref = min(full_ref_coords, key=lambda ref: abs(coord[0] - ref[0]) + abs(coord[1] - ref[1]))
            reference_coords_final.append(best_ref)

    # For overlapping target region → sample from whole reference patch pool
    for coord in overlap_target_coords:
        target_coords_final.append(coord)
        if full_ref_coords:
            # Find closest patch in full reference pool
            best_ref = min(full_ref_coords, key=lambda ref: abs(coord[0] - ref[0]) + abs(coord[1] - ref[1]))
            reference_coords_final.append(best_ref)
    return target_coords_final, reference_coords_final

def strategy_2(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_H = H
    patch_W = W
    patch_w = patch_W // patch_size
    patch_h = patch_H // patch_size
    mask_A_patch_coords = [tuple(coord) for coord in data['mask_A_patch_coords']]
    mask_B_patch_coords = [tuple(coord) for coord in data['mask_B_patch_coords']]
    overlap_patch_coords = [tuple(coord) for coord in data['overlap_patch_coords']]
    mask_A_patch_set = set(mask_A_patch_coords)
    mask_B_patch_set = set(mask_B_patch_coords)
    overlap_patch_set = set(overlap_patch_coords)

    target_patch_set = set(overlap_patch_coords)
    for oy, ox in overlap_patch_coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if abs(dy) + abs(dx) <= 1:  # Manhattan distance <= 1
                    ny, nx = oy + dy, ox + dx
                    if 0 <= ny < patch_h and 0 <= nx < patch_w:
                        target_patch_set.add((ny, nx))
    # Ensure all target patches are foreground patches (part of either entity A or B)
    foreground_patch_set = mask_A_patch_set | mask_B_patch_set
    target_patch_set = target_patch_set & foreground_patch_set
    target_patch_coords = list(target_patch_set)

    # Step 5: Partition target region into three parts
    A_only_coords = [coord for coord in target_patch_coords if coord in mask_A_patch_set and coord not in overlap_patch_set]
    B_only_coords = [coord for coord in target_patch_coords if coord in mask_B_patch_set and coord not in overlap_patch_set]
    overlap_target_coords = [coord for coord in target_patch_coords if coord in overlap_patch_set]
    k_seeds = 5
    band_R = 2
    seed_rng = 12345
    allow_self_region = False
    avoid_identity_ref = True
    rng = random.Random(seed_rng)
    """
    Strategy:
      1) Seeds from overlap_target_coords via FPS (k_seeds).
      2) Band = L1<=band_R around overlap_target_coords ∩ foreground (A∪B).
      3) Voronoi assign band targets to nearest seed.
      4) For each region, randomly pick a donor region (optionally excluding itself).
      5) For each target in the region, randomly sample a reference from the donor region's patches
         (optionally avoid identity).

    Returns:
      target_coords_final, reference_coords_final, meta
    """
    rng = random.Random(seed_rng)

    def _manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _farthest_point_sampling(points: List[Tuple[int,int]], k: int) -> List[Tuple[int,int]]:
        if not points:
            return []
        pts = list(points)
        cy = sum(p[0] for p in pts) / len(pts)
        cx = sum(p[1] for p in pts) / len(pts)
        start = min(pts, key=lambda p: (p[0]-cy)**2 + (p[1]-cx)**2)
        seeds = [start]
        if k <= 1 or len(pts) == 1:
            return seeds
        min_d = {p: _manhattan(p, start) for p in pts}
        for _ in range(1, min(k, len(pts))):
            nxt = max(pts, key=lambda p: min_d[p])
            seeds.append(nxt)
            for p in pts:
                d = _manhattan(p, nxt)
                if d < min_d[p]:
                    min_d[p] = d
        return seeds

    def _build_band_around_coords(centers: List[Tuple[int,int]], foreground: set, band_R: int) -> List[Tuple[int,int]]:
        if not centers:
            return []
        ys = [y for y, _ in centers]; xs = [x for _, x in centers]
        y0 = max(0, min(ys) - band_R); y1 = min(patch_h, max(ys) + band_R + 1)
        x0 = max(0, min(xs) - band_R); x1 = min(patch_w, max(xs) + band_R + 1)
        out = []
        centers_set = set(centers)
        for py in range(y0, y1):
            for px in range(x0, x1):
                if (py, px) not in foreground: 
                    continue
                # within band of any center
                for cy, cx in centers_set:
                    if abs(py - cy) + abs(px - cx) <= band_R:
                        out.append((py, px))
                        break
        return out

    foreground = mask_A_patch_set | mask_B_patch_set

    # 1) Seeds
    seeds = _farthest_point_sampling(overlap_target_coords, k=k_seeds)
    if not seeds:
        return [], [], {"reason": "no seeds (empty overlap_target_coords)"}

    # 2) Band
    band_targets = _build_band_around_coords(overlap_target_coords, foreground, band_R)
    if not band_targets:
        return [], [], {"reason": "empty band around overlap"}

    # 3) Voronoi regions
    regions: Dict[Tuple[int,int], List[Tuple[int,int]]] = {s: [] for s in seeds}
    for p in band_targets:
        s_best = min(seeds, key=lambda s: _manhattan(p, s))
        regions[s_best].append(p)

    # Filter to non-empty regions and fix order
    seed_list = [s for s in seeds if regions[s]]
    if not seed_list:
        return [], [], {"reason": "all regions empty after assignment"}

    # 4) For each region, randomly pick a donor region
    target_coords_final: List[Tuple[int,int]] = []
    reference_coords_final: List[Tuple[int,int]] = []
    region_meta = []

    for s in seed_list:
        own = regions[s]
        if not own:
            continue
        # candidate donor regions
        if allow_self_region:
            candidates = seed_list
        else:
            candidates = [t for t in seed_list if t != s]
            if not candidates:
                # if only one region and self not allowed, fallback to self
                candidates = [s]
        donor_seed = rng.choice(candidates)
        donor_pool = regions[donor_seed]

        if not donor_pool:
            # extremely rare (shouldn't happen after filtering), skip
            continue

        # 5) For each target in region, sample random donor patch
        for p in own:
            if avoid_identity_ref:
                # avoid picking exact same patch if donor==self and pool size > 1
                if donor_seed == s and len(donor_pool) > 1:
                    # pick until different (bounded tries)
                    for _ in range(5):
                        q = rng.choice(donor_pool)
                        if q != p:
                            break
                    else:
                        q = rng.choice(donor_pool)
                else:
                    q = rng.choice(donor_pool)
            else:
                q = rng.choice(donor_pool)

            target_coords_final.append(p)
            reference_coords_final.append(q)

        region_meta.append({
            "seed": s,
            "region_size": len(own),
            "donor_seed": donor_seed,
            "donor_pool_size": len(donor_pool)
        })
    return target_coords_final, reference_coords_final

def strategy_3(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]
    
    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size
    
    # Convert to sets of (y, x) tuples
    mask_A_patch_coords = [tuple(coord) for coord in data['mask_A_patch_coords']]
    mask_B_patch_coords = [tuple(coord) for coord in data['mask_B_patch_coords']]
    overlap_patch_coords = [tuple(coord) for coord in data['overlap_patch_coords']]
    
    mask_A_patch_set = set(mask_A_patch_coords)
    mask_B_patch_set = set(mask_B_patch_coords)
    overlap_patch_set = set(overlap_patch_coords)
    
    # Define overlap_target_coords
    overlap_target_coords = list(overlap_patch_set)
    
    if not overlap_target_coords:
        return [], []
    
    # Create A_only and B_only coordinate pools
    A_only_coords = [coord for coord in mask_A_patch_coords if coord not in overlap_patch_set]
    B_only_coords = [coord for coord in mask_B_patch_coords if coord not in overlap_patch_set]
    
    rng = random.Random(2025)
    
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _farthest_point_sampling(points: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
        if not points or k <= 0:
            return []
        
        pts = list(points)
        if k >= len(pts):
            return pts
        
        # Start with centroid-closest point
        cy = sum(p[0] for p in pts) / len(pts)
        cx = sum(p[1] for p in pts) / len(pts)
        start = min(pts, key=lambda p: (p[0] - cy)**2 + (p[1] - cx)**2)
        
        seeds = [start]
        if k <= 1:
            return seeds
        
        # Track minimum distances to existing seeds
        min_distances = {p: _manhattan(p, start) for p in pts}
        
        for _ in range(1, k):
            # Pick point with maximum minimum distance to existing seeds
            next_seed = max(pts, key=lambda p: min_distances[p])
            seeds.append(next_seed)
            
            # Update minimum distances
            for p in pts:
                dist = _manhattan(p, next_seed)
                if dist < min_distances[p]:
                    min_distances[p] = dist
        
        return seeds
    
    def _nearest(target: Tuple[int, int], candidates: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not candidates:
            return target  # Fallback
        return min(candidates, key=lambda c: _manhattan(target, c))
    
    # Select k seeds using farthest-point sampling
    k = min(5, len(overlap_target_coords))  # Reasonable default
    seeds = _farthest_point_sampling(overlap_target_coords, k)
    
    if not seeds:
        return [], []
    
    # Assign each overlap patch to nearest seed (Voronoi partition)
    regions = {seed: [] for seed in seeds}
    for patch in overlap_target_coords:
        nearest_seed = min(seeds, key=lambda s: _manhattan(patch, s))
        regions[nearest_seed].append(patch)
    
    target_coords_final = []
    reference_coords_final = []
    
    # For each region
    for seed in seeds:
        region_patches = regions[seed]
        if not region_patches:
            continue
        
        # Randomly assign reference pool: A_only_coords or B_only_coords
        if A_only_coords and B_only_coords:
            reference_pool = rng.choice([A_only_coords, B_only_coords])
        elif A_only_coords:
            reference_pool = A_only_coords
        elif B_only_coords:
            reference_pool = B_only_coords
        else:
            # Fallback to general pool if both A_only and B_only are empty
            reference_pool = mask_A_patch_coords + mask_B_patch_coords
        
        # For each patch in the region, find nearest reference
        for target_patch in region_patches:
            target_coords_final.append(target_patch)
            nearest_ref = _nearest(target_patch, reference_pool)
            reference_coords_final.append(nearest_ref)
    
    return target_coords_final, reference_coords_final

def strategy_4(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]
    
    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size
    
    # Build sets
    A_set = set(tuple(coord) for coord in data['mask_A_patch_coords'])
    B_set = set(tuple(coord) for coord in data['mask_B_patch_coords'])
    overlap_set = set(tuple(coord) for coord in data['overlap_patch_coords'])
    foreground_set = A_set | B_set
    
    # A_only and B_only regions
    A_only = A_set - overlap_set
    B_only = B_set - overlap_set
    
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
        
        # Start with centroid-closest point
        cy = sum(p[0] for p in pts) / len(pts)
        cx = sum(p[1] for p in pts) / len(pts)
        start = min(pts, key=lambda p: (p[0] - cy)**2 + (p[1] - cx)**2)
        
        seeds = [start]
        if k <= 1:
            return seeds
        
        # Track minimum distances to existing seeds
        min_distances = {p: _manhattan(p, start) for p in pts}
        
        for _ in range(1, k):
            # Pick point with maximum minimum distance to existing seeds
            next_seed = max(pts, key=lambda p: min_distances[p])
            seeds.append(next_seed)
            
            # Update minimum distances
            for p in pts:
                dist = _manhattan(p, next_seed)
                if dist < min_distances[p]:
                    min_distances[p] = dist
        
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

    # Choose seeds from overlap_set using farthest-point sampling
    overlap_list = list(overlap_set)
    seeds = _farthest_point_sampling(overlap_list, k_seeds)
    
    if not seeds:
        return [], []

    # Partition band into Voronoi regions
    regions = {seed: [] for seed in seeds}
    for patch in band_targets:
        nearest_seed = min(seeds, key=lambda s: _manhattan(patch, s))
        regions[nearest_seed].append(patch)
    
    # Generate candidate offsets ordered by L1 distance (1 ≤ L1 ≤ 4)
    candidate_offsets = []
    for l1_dist in range(1, 5):  # 1 ≤ L1 ≤ 4
        for dy in range(-l1_dist, l1_dist + 1):
            for dx in range(-l1_dist, l1_dist + 1):
                if _manhattan((0, 0), (dy, dx)) == l1_dist:
                    candidate_offsets.append((dy, dx))
    
    target_coords_final = []
    reference_coords_final = []
    
    # Process each region
    for seed in seeds:
        region_patches = regions[seed]
        if not region_patches:
            continue
        
        valid_offset = None
        chosen_entity = None
        fallback_used = None
        
        # Try each candidate offset
        for dy, dx in candidate_offsets:
            shifted_region = [(py + dy, px + dx) for py, px in region_patches]
            
            # Validate bounds
            if all(0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted_region):
                shifted_set = set(shifted_region)
                
                # Validate entity: must lie entirely in A_only or entirely in B_only
                if shifted_set.issubset(A_only):
                    valid_offset = (dy, dx)
                    chosen_entity = 'A_only'
                    break
                elif shifted_set.issubset(B_only):
                    valid_offset = (dy, dx)
                    chosen_entity = 'B_only'
                    break
        
        # Fallback 1: allow shifted region ⊆ foreground_set
        if valid_offset is None:
            for dy, dx in candidate_offsets:
                shifted_region = [(py + dy, px + dx) for py, px in region_patches]
                
                # Validate bounds
                if all(0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted_region):
                    shifted_set = set(shifted_region)
                    
                    if shifted_set.issubset(foreground_set):
                        valid_offset = (dy, dx)
                        chosen_entity = 'foreground'
                        fallback_used = 'foreground_subset'
                        break
        
        # Fallback 2: map each patch individually to nearest opposite-entity patch
        if valid_offset is None:
            fallback_used = 'individual_nearest'
            for py, px in region_patches:
                target_coords_final.append((py, px))
                
                # Find nearest patch in opposite entity
                if (py, px) in A_set and B_only:
                    nearest_ref = min(B_only, key=lambda ref: _manhattan((py, px), ref))
                elif (py, px) in B_set and A_only:
                    nearest_ref = min(A_only, key=lambda ref: _manhattan((py, px), ref))
                else:
                    # Fallback to any foreground patch
                    foreground_list = list(foreground_set - {(py, px)})
                    if foreground_list:
                        nearest_ref = min(foreground_list, key=lambda ref: _manhattan((py, px), ref))
                    else:
                        nearest_ref = (py, px)  # Last resort
                
                reference_coords_final.append(nearest_ref)
        else:
            # Apply valid offset to all patches in region
            dy, dx = valid_offset
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append((py + dy, px + dx))
    
    return target_coords_final, reference_coords_final

def strategy_5(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Build sets
    A_set = set(tuple(coord) for coord in data['mask_A_patch_coords'])
    B_set = set(tuple(coord) for coord in data['mask_B_patch_coords'])
    overlap_set = set(tuple(coord) for coord in data['overlap_patch_coords'])
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
    seeds = _farthest_point_sampling(list(overlap_set), k_seeds)
    if not seeds:
        return [], []

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

    for seed in seeds:
        region_patches = regions[seed]
        if not region_patches:
            continue

        # --- Primary attempt: a single offset that maps the entire region
        #     into A_only OR entirely into B_only (i.e., never overlap).
        valid_offset = None
        for dy, dx in candidate_offsets:
            shifted = [(py + dy, px + dx) for (py, px) in region_patches]
            # bounds
            if not all(0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted):
                continue
            shifted_set = set(shifted)
            if shifted_set.issubset(A_only) or shifted_set.issubset(B_only):
                valid_offset = (dy, dx)
                break

        if valid_offset is not None:
            dy, dx = valid_offset
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append((py + dy, px + dx))
            continue

        # --- New fallback: choose the offset that maps the MAX number of patches
        #     into non-overlap (any entity, but NOT overlap), apply to those;
        #     for the remainder, map to nearest non-overlap patch.
        best_offset = None
        best_mapped_count = -1
        best_mapped_mask = None  # list[bool] parallel to region_patches

        for dy, dx in candidate_offsets:
            shifted = [(py + dy, px + dx) for (py, px) in region_patches]
            # bounds mask
            in_bounds = [0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted]
            # non-overlap mask (also implies foreground)
            non_overlap_mask = [
                in_bounds[i] and (shifted[i] in non_overlap_foreground)
                for i in range(len(shifted))
            ]
            mapped_count = sum(non_overlap_mask)
            if mapped_count > best_mapped_count:
                best_mapped_count = mapped_count
                best_offset = (dy, dx)
                best_mapped_mask = non_overlap_mask

        if best_offset is not None and best_mapped_count > 0:
            dy, dx = best_offset
            for i, (py, px) in enumerate(region_patches):
                target_coords_final.append((py, px))
                if best_mapped_mask[i]:
                    # Use offset-mapped non-overlap reference
                    reference_coords_final.append((py + dy, px + dx))
                else:
                    # Fill with nearest non-overlap reference (never overlap)
                    reference_coords_final.append(_nearest_non_overlap((py, px)))
        else:
            # Nothing could be offset into non-overlap → all via nearest non-overlap
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append(_nearest_non_overlap((py, px)))

    return target_coords_final, reference_coords_final

def strategy_6(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Build sets
    A_set = set(tuple(coord) for coord in data['mask_A_patch_coords'])
    B_set = set(tuple(coord) for coord in data['mask_B_patch_coords'])
    overlap_set = set(tuple(coord) for coord in data['overlap_patch_coords'])
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
    seeds = _farthest_point_sampling(list(overlap_set), k_seeds)
    if not seeds:
        return [], []

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

    for seed in seeds:
        region_patches = regions[seed]
        if not region_patches:
            continue

        # --- Primary attempt: a single offset that maps the entire region
        #     into A_only OR entirely into B_only (i.e., never overlap).
        valid_offset = None
        for dy, dx in candidate_offsets:
            shifted = [(py + dy, px + dx) for (py, px) in region_patches]
            # bounds
            if not all(0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted):
                continue
            shifted_set = set(shifted)
            if shifted_set.issubset(A_only) or shifted_set.issubset(B_only):
                valid_offset = (dy, dx)
                break

        if valid_offset is not None:
            dy, dx = valid_offset
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append((py + dy, px + dx))
            continue

        # --- Fallback: choose the offset that maps the MAX number of patches
        #     into non-overlap (any entity, but NOT overlap), apply to those;
        #     for the remainder, map to nearest non-overlap patch.
        best_offset = None
        best_mapped_count = -1
        best_mapped_mask = None  # list[bool] parallel to region_patches

        for dy, dx in candidate_offsets:
            shifted = [(py + dy, px + dx) for (py, px) in region_patches]
            # bounds mask
            in_bounds = [0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted]
            # non-overlap mask (also implies foreground)
            non_overlap_mask = [
                in_bounds[i] and (shifted[i] in non_overlap_foreground)
                for i in range(len(shifted))
            ]
            mapped_count = sum(non_overlap_mask)
            if mapped_count > best_mapped_count:
                best_mapped_count = mapped_count
                best_offset = (dy, dx)
                best_mapped_mask = non_overlap_mask

        if best_offset is not None and best_mapped_count > 0:
            dy, dx = best_offset
            for i, (py, px) in enumerate(region_patches):
                target_coords_final.append((py, px))
                if best_mapped_mask[i]:  # pyright: ignore[reportOptionalSubscript]
                    # Use offset-mapped non-overlap reference
                    reference_coords_final.append((py + dy, px + dx))
                else:
                    # Fill with nearest non-overlap reference (never overlap)
                    reference_coords_final.append(_nearest_non_overlap((py, px)))
        else:
            # Nothing could be offset into non-overlap → all via nearest non-overlap
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append(_nearest_non_overlap((py, px)))

    # --- Symmetric augmentation: add reversed mappings (ref -> target)
    # Avoid duplicate target patches.
    existing_targets = set(target_coords_final)
    original_pairs = list(zip(target_coords_final, reference_coords_final))
    for tgt, ref in original_pairs:
        if ref not in existing_targets:
            target_coords_final.append(ref)
            reference_coords_final.append(tgt)
            existing_targets.add(ref)
    num_new_targets = len(target_coords_final) - len(original_pairs)
    ratio_new_targets = num_new_targets / len(target_coords_final) if target_coords_final else 0
    print(f"Ratio of newly added targets: {ratio_new_targets:.4f}")

    return target_coords_final, reference_coords_final

def strategy_7(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Build sets
    A_set = set(tuple(coord) for coord in data['mask_A_patch_coords'])
    B_set = set(tuple(coord) for coord in data['mask_B_patch_coords'])
    overlap_set = set(tuple(coord) for coord in data['overlap_patch_coords'])
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

    return target_coords_final, reference_coords_final

def strategy_8(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Build sets
    A_set = set(tuple(coord) for coord in data['mask_A_patch_coords'])
    B_set = set(tuple(coord) for coord in data['mask_B_patch_coords'])
    overlap_set = set(tuple(coord) for coord in data['overlap_patch_coords'])
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
    seeds = _farthest_point_sampling(list(overlap_set), k_seeds)
    if not seeds:
        return [], []

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

    for seed in seeds:
        region_patches = regions[seed]
        if not region_patches:
            continue

        # --- Primary attempt: a single offset that maps the entire region
        #     into A_only OR entirely into B_only (i.e., never overlap).
        valid_offset = None
        for dy, dx in candidate_offsets:
            shifted = [(py + dy, px + dx) for (py, px) in region_patches]
            # bounds check
            if not all(0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted):
                continue
            shifted_set = set(shifted)
            
            # NEW: Check if shifted_set overlaps with band_targets - skip if it does
            if shifted_set & band_targets:
                continue
                
            # Check if shifted_set is entirely in A_only or B_only
            if shifted_set.issubset(A_only) or shifted_set.issubset(B_only):
                valid_offset = (dy, dx)
                break

        if valid_offset is not None:
            dy, dx = valid_offset
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append((py + dy, px + dx))
            continue

        # --- Fallback: choose the offset that maps the MAX number of patches
        #     into non-overlap (any entity, but NOT overlap), apply to those;
        #     for the remainder, map to nearest non-overlap patch.
        best_offset = None
        best_mapped_count = -1
        best_mapped_mask = None  # list[bool] parallel to region_patches

        for dy, dx in candidate_offsets:
            shifted = [(py + dy, px + dx) for (py, px) in region_patches]
            # bounds mask
            in_bounds = [0 <= ny < patch_h and 0 <= nx < patch_w for ny, nx in shifted]
            # non-overlap mask (also implies foreground)
            non_overlap_mask = [
                in_bounds[i] and (shifted[i] in non_overlap_foreground)
                for i in range(len(shifted))
            ]
            mapped_count = sum(non_overlap_mask)
            if mapped_count > best_mapped_count:
                best_mapped_count = mapped_count
                best_offset = (dy, dx)
                best_mapped_mask = non_overlap_mask

        if best_offset is not None and best_mapped_count > 0:
            dy, dx = best_offset
            for i, (py, px) in enumerate(region_patches):
                target_coords_final.append((py, px))
                if best_mapped_mask[i]:  # pyright: ignore[reportOptionalSubscript]
                    # Use offset-mapped non-overlap reference
                    reference_coords_final.append((py + dy, px + dx))
                else:
                    # Fill with nearest non-overlap reference (never overlap)
                    reference_coords_final.append(_nearest_non_overlap((py, px)))
        else:
            # Nothing could be offset into non-overlap → all via nearest non-overlap
            for py, px in region_patches:
                target_coords_final.append((py, px))
                reference_coords_final.append(_nearest_non_overlap((py, px)))

    # --- Symmetric augmentation: add reversed mappings (ref -> target)
    # Avoid duplicate target patches.
    existing_targets = set(target_coords_final)
    original_pairs = list(zip(target_coords_final, reference_coords_final))
    for tgt, ref in original_pairs:
        if ref not in existing_targets:
            target_coords_final.append(ref)
            reference_coords_final.append(tgt)
            existing_targets.add(ref)
    num_new_targets = len(target_coords_final) - len(original_pairs)
    ratio_new_targets = num_new_targets / len(target_coords_final) if target_coords_final else 0
    print(f"Ratio of newly added targets: {ratio_new_targets:.4f}")

    return target_coords_final, reference_coords_final
def strategy_9(loaded_data) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    data = loaded_data['step3_data_list'][0]
    H, W = loaded_data['img_array'].shape[0], loaded_data['img_array'].shape[1]

    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Build sets
    A_set = set(tuple(coord) for coord in data['mask_A_patch_coords'])
    B_set = set(tuple(coord) for coord in data['mask_B_patch_coords'])
    overlap_set = set(tuple(coord) for coord in data['overlap_patch_coords'])
    foreground_set = A_set | B_set

    # Non-overlap pools
    A_only = A_set - overlap_set
    B_only = B_set - overlap_set
    non_overlap_foreground = foreground_set - overlap_set

    if not overlap_set:
        return [], []

    band_R = 1
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

    # --- Symmetric augmentation: add reversed mappings (ref -> target)
    # Avoid duplicate target patches.
    existing_targets = set(target_coords_final)
    original_pairs = list(zip(target_coords_final, reference_coords_final))
    for tgt, ref in original_pairs:
        if ref not in existing_targets:
            target_coords_final.append(ref)
            reference_coords_final.append(tgt)
            existing_targets.add(ref)
    num_new_targets = len(target_coords_final) - len(original_pairs)
    ratio_new_targets = num_new_targets / len(target_coords_final) if target_coords_final else 0
    print(f"Ratio of newly added targets: {ratio_new_targets:.4f}")

    return target_coords_final, reference_coords_final
