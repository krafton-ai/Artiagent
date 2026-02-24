import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def patch_coor_to_ind(x, y, w, txt_len):
    return y * w + x + txt_len

def patch_ind_to_coor(ind, w, txt_len, return_shape=False):
    ind -= txt_len
    y = ind // w
    x = ind % w
    return [y, x]

def patch_indices_to_coords(indices, w, txt_len=512):
    """
    Convert patch indices back to patch coordinates.
    
    Args:
        indices: List of patch indices
        w: Patch width
        txt_len: Text length offset
    
    Returns:
        List of (y, x) patch coordinates
    """
    return [patch_ind_to_coor(ind, w, txt_len) for ind in indices]

def bbox_to_patch_indices(bbox_coordinates, h, w, patch_size=16, txt_len=512):
    xmin, xmax, ymin, ymax = bbox_coordinates
    patch_xmin = xmin // patch_size
    patch_xmax = (xmax - 1) // patch_size
    patch_ymin = ymin // patch_size
    patch_ymax = (ymax - 1) // patch_size
    indices = [
        patch_coor_to_ind(px, py, w, txt_len)
        for py in range(patch_ymin, patch_ymax + 1)
        for px in range(patch_xmin, patch_xmax + 1)
    ]
    return indices

def bbox_to_patch_coords(bbox_coordinates, patch_size=16, return_shape=False):
    xmin, xmax, ymin, ymax = bbox_coordinates
    patch_xmin = xmin // patch_size
    patch_xmax = (xmax - 1) // patch_size
    patch_ymin = ymin // patch_size
    patch_ymax = (ymax - 1) // patch_size
    coords = [ (py, px)
        for py in range(patch_ymin, patch_ymax + 1)
        for px in range(patch_xmin, patch_xmax + 1)
    ]
    if return_shape:
        return coords, patch_ymax-patch_ymin+1, patch_xmax-patch_xmin+1
    return coords

# New shape-based functions
def mask_to_patch_indices(mask, patch_size=16, txt_len=512):
    """
    Convert a binary mask to patch indices.
    
    Args:
        mask: Binary mask (numpy array) where 1 indicates the region of interest
        patch_size: Size of each patch (default 16 for Flux)
        txt_len: Text length offset (default 512)
    
    Returns:
        List of patch indices
    """
    h, w = mask.shape
    patch_h, patch_w = h // patch_size, w // patch_size
    
    # Downsample mask to patch resolution
    patch_mask = np.zeros((patch_h, patch_w), dtype=bool)
    
    for py in range(patch_h):
        for px in range(patch_w):
            # Get the patch region in the original mask
            y_start, y_end = py * patch_size, (py + 1) * patch_size
            x_start, x_end = px * patch_size, (px + 1) * patch_size
            
            # If any part of the patch overlaps with the mask, include it
            patch_region = mask[y_start:y_end, x_start:x_end]
            if np.any(patch_region):
                patch_mask[py, px] = True
    
    # Convert patch coordinates to indices
    indices = []
    for py in range(patch_h):
        for px in range(patch_w):
            if patch_mask[py, px]:
                indices.append(patch_coor_to_ind(px, py, patch_w, txt_len))
    
    return indices

def mask_to_patch_coords(mask, patch_size=16):
    """
    Convert a binary mask to patch coordinates.
    
    Args:
        mask: Binary mask (numpy array) where 1 indicates the region of interest
        patch_size: Size of each patch (default 16 for Flux)
    
    Returns:
        List of (py, px) patch coordinates
    """
    h, w = mask.shape
    patch_h, patch_w = h // patch_size, w // patch_size
    
    # Downsample mask to patch resolution
    patch_mask = np.zeros((patch_h, patch_w), dtype=bool)
    
    for py in range(patch_h):
        for px in range(patch_w):
            # Get the patch region in the original mask
            y_start, y_end = py * patch_size, (py + 1) * patch_size
            x_start, x_end = px * patch_size, (px + 1) * patch_size
            
            # If any part of the patch overlaps with the mask, include it
            patch_region = mask[y_start:y_end, x_start:x_end]
            if np.any(patch_region):
                patch_mask[py, px] = True
    
    # Convert patch coordinates to list
    coords = []
    for py in range(patch_h):
        for px in range(patch_w):
            if patch_mask[py, px]:
                coords.append((py, px))
    
    return coords

def get_closest_patch_ind(h, w, bbox_coordinates, patch_size=16, txt_len=512):
    # Get only valid coordinates (value == 1)

    large_array = np.ones((h,w), dtype=int)
    small_grid_coords, bbox_h, bbox_w = bbox_to_patch_coords(bbox_coordinates, patch_size=patch_size, return_shape=True)

    for y, x in small_grid_coords:
        large_array[y,x] = 0

    valid_coords = np.argwhere(large_array == 1)
    
    result = np.empty((bbox_h, bbox_w), dtype=object)
    min_h, min_w = small_grid_coords[0]
    
    for idx, coord in enumerate(small_grid_coords):
        y, x = coord
        distances = np.abs(valid_coords[:, 0] - y) + np.abs(valid_coords[:, 1] - x)
        min_idx = np.argmin(distances)
        closest_coord = tuple(valid_coords[min_idx])
        result[y-min_h,x-min_w] = closest_coord
    
    return [patch_coor_to_ind(x,y,w,txt_len) for y, x in result.flatten()]

def get_neighbors_patch_ind(h, w, bbox_coordinates, img_ids, patch_size=16, txt_len=512):
    # Get the patch coordinates and shape of the bbox
    small_grid_coords, bbox_h, bbox_w = bbox_to_patch_coords(bbox_coordinates, patch_size=patch_size, return_shape=True)
    small_grid_coords = np.array(small_grid_coords)

    indices_array = img_ids.cpu().numpy().squeeze().copy()
    indices_array = indices_array.reshape((h*w,-1))

    # indices_array = indices_array.reshape((h*w,-1))

    # Compute bbox center in patch coordinates
    min_h, min_w = np.min(small_grid_coords, axis=0)
    max_h, max_w = np.max(small_grid_coords, axis=0)
    center_y = (min_h + max_h) // 2
    center_x = (min_w + max_w) // 2

    min_h_p, min_w_p = max(min_h-2, 0), max(min_w-2, 0)
    max_h_p, max_w_p = min(h, max_h+2), min(w, max_w+2)

    # Compute shortest distance from center to bbox edge
    radius_y = min(center_y - min_h, max_h - center_y)
    radius_x = min(center_x - min_w, max_w - center_x)
    radius = min(radius_y, radius_x)

    # Get all valid coordinates in the grid
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    all_coords = np.stack([yy.ravel(), xx.ravel()], axis=1)

    # Exclude bbox coordinates
    bbox_set = set(map(tuple, small_grid_coords))
    filtered_coords = [tuple(coord) for coord in all_coords if tuple(coord) not in bbox_set]

    # result_coords=[]
    # for center_y, center_x in small_grid_coords:
    #     neighbors = []
    #     for dy in range(-radius-1, radius + 2):
    #         for dx in range(-radius-1, radius + 2):
    #             if abs(dy) + abs(dx) <= radius+2:
    #                 ny, nx = center_y + dy, center_x + dx
    #                 if 0 <= ny < h and 0 <= nx < w:
    #                     if (ny, nx) not in bbox_set:
    #                         neighbors.append((ny, nx))
    #     result_coords.append(neighbors)

    # return small_grid_coords.tolist(), result_coords

    result=[]
    for center_y, center_x in small_grid_coords:
        neighbors = []
        for dy in range(-radius-1, radius + 2):
            for dx in range(-radius-1, radius + 2):
                if abs(dy) + abs(dx) <= radius+1:
                    ny, nx = center_y + dy, center_x + dx
                    if min_h_p <= ny < max_h_p and min_w_p <= nx < max_w_p:
                        if (ny, nx) not in bbox_set:
                            neighbors.append((ny, nx))
        # if len(neighbors) == 0:
            # import pdb;pdb.set_trace()
        neighbors_ind = indices_array[[patch_coor_to_ind(x,y,w,0) for (y,x) in neighbors],:]        
        result.append(neighbors_ind.mean(0))

    return torch.from_numpy(np.array(result)).unsqueeze(0)

def perturb_pe(h, w, bbox_coordinates, img_ids, patch_size=16, txt_len=512):
    patch_ids = bbox_to_patch_indices(bbox_coordinates, h, w, patch_size, txt_len=0)
    indices_array = torch.from_numpy(img_ids.cpu().numpy().squeeze().copy()[patch_ids, :])
    noise = torch.randn_like(indices_array)
    # Mask for non-zero elements
    nonzero_mask = indices_array != 0
    # Clone indices_array to preserve original
    perturbed_indices_array = indices_array.clone()
    # Apply noise only to non-zero elements
    perturbed_indices_array[nonzero_mask] += noise[nonzero_mask]
    # perturbed_indices_array = indices_array + torch.randn_like(indices_array) * 0.3
    return perturbed_indices_array.unsqueeze(0)

def shuffle_pe(h, w, patch_ids, patch_size=16, txt_len=512, intensity=3):
    bbox_coords = [patch_ind_to_coor(ind, w, txt_len) for ind in patch_ids]
    # bbox_coords = bbox_to_patch_coords(bbox_coordinates, patch_size=patch_size)

    shuffled_coords = []

    for y, x in bbox_coords:
        dx = torch.randint(-intensity, intensity + 1, (1,)).item()
        dy = torch.randint(-intensity, intensity + 1, (1,)).item()
        # dx = 0
        # dy = 0

        new_x = max(0, min(x + dx, w - 1))
        new_y = max(0, min(y + dy, h - 1))

        shuffled_coords.append((new_y, new_x))

    return [patch_coor_to_ind(x,y,w,txt_len) for y, x in shuffled_coords]

def sample_closest_patch_ind(h, w, patch_indices, reference_patch_indices, patch_size=16, txt_len=512):
    """
    Sample closest patches for arbitrary shape with randomization.
    
    Args:
        h, w: Patch grid dimensions  
        patch_indices: List of patch indices defining the shape
        reference_patch_indices: List of patch indices to use as reference/candidates
        patch_size: Size of each patch
        txt_len: Text length offset
    
    Returns:
        List of sampled closest patch indices
    """
    shape_coords = np.array([patch_ind_to_coor(ind, w, txt_len) for ind in patch_indices])
    
    # Convert reference patch indices to coordinates
    reference_coords = np.array([patch_ind_to_coor(ind, w, txt_len) for ind in reference_patch_indices])
    
    result = []
    for y, x in shape_coords:
        if 0 <= y < h and 0 <= x < w:
            # Calculate distances only to reference patch coordinates
            distances = np.abs(reference_coords[:, 0] - y) + np.abs(reference_coords[:, 1] - x)
            inv_d = 1.0 / (distances + 1e-8)
            inv_d = np.pow(inv_d, 2)
            p_weight = inv_d / np.sum(inv_d)
            idx = np.random.choice(len(distances), p=p_weight)
            closest_coord = tuple(reference_coords[idx])
            result.append(patch_coor_to_ind(closest_coord[1], closest_coord[0], w, txt_len))
    
    return result

def get_closest_patch_coords(target_coords, reference_coords):
    """
    Map each target coordinate to its closest reference coordinate.
    
    Args:
        target_coords: List of (y, x) coordinates that need to be mapped
        reference_coords: List of (y, x) coordinates to use as reference/candidates
    
    Returns:
        List of closest reference coordinates for each target coordinate
    """
    target_coords = np.array(target_coords)
    reference_coords = np.array(reference_coords)
    
    result = []
    for ty, tx in target_coords:
        # Calculate Manhattan distances to all reference coordinates
        distances = np.abs(reference_coords[:, 0] - ty) + np.abs(reference_coords[:, 1] - tx)
        min_idx = np.argmin(distances)
        closest_coord = tuple(reference_coords[min_idx])
        result.append(closest_coord)
    
    return result


def get_closest_patch_inds(h, w, target_patch_indices, reference_patch_indices, txt_len=512):
    """
    Map each target patch index to the closest reference patch index using Manhattan distance.
    
    Args:
        h, w: Patch grid dimensions (not used directly but kept for API symmetry)
        target_patch_indices: List of patch indices to map
        reference_patch_indices: List of candidate reference patch indices
        txt_len: Text length offset used in index<->coord conversions
    
    Returns:
        List of closest reference patch indices corresponding to each target patch index
    """
    if len(target_patch_indices) == 0 or len(reference_patch_indices) == 0:
        return []

    # Convert indices to (y, x) coordinates
    target_coords = np.array([patch_ind_to_coor(ind, w, txt_len) for ind in target_patch_indices])
    reference_coords = np.array([patch_ind_to_coor(ind, w, txt_len) for ind in reference_patch_indices])

    result = []
    for ty, tx in target_coords:
        distances = np.abs(reference_coords[:, 0] - ty) + np.abs(reference_coords[:, 1] - tx)
        min_idx = int(np.argmin(distances))
        result.append(reference_patch_indices[min_idx])

    return result
