import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def patch_coor_to_ind(x, y, w, txt_len):
    return y * w + x + txt_len

def patch_ind_to_coor(ind, w, txt_len, return_shape=False):
    ind -= txt_len
    y = ind // w
    x = ind % w
    return [y, x]

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


# def sample_closest_patch_ind(h, w, patch_ids, patch_size=16, txt_len=512):
#     # Get only valid coordinates (value == 1)

#     small_grid_coords = np.array([patch_ind_to_coor(ind, w, txt_len) for ind in patch_ids])
#     # import pdb;pdb.set_trace()
#     # Separate x and y coordinates
#     y_coords = small_grid_coords[:, 0]
#     x_coords = small_grid_coords[:, 1]

#     # Compute width and height as ranges
#     bbox_w = x_coords.max() - x_coords.min()+1
#     bbox_h = y_coords.max() - y_coords.min()+1

#     large_array = np.ones((h,w), dtype=int)
#     # small_grid_coords, bbox_h, bbox_w = bbox_to_patch_coords(bbox_coordinates, patch_size=patch_size, return_shape=True)

#     for y, x in small_grid_coords:
#         large_array[y,x] = 0

#     valid_coords = np.argwhere(large_array == 1)
    
#     result = np.empty((bbox_h, bbox_w), dtype=object)
#     min_h, min_w = small_grid_coords[0]
    
#     for idx, coord in enumerate(small_grid_coords):
#         y, x = coord
#         distances = np.abs(valid_coords[:, 0] - y) + np.abs(valid_coords[:, 1] - x)
#         # distances = np.sqrt((valid_coords[:, 0] - y)**2 + (valid_coords[:, 1] - x)**2)
#         inv_d = 1.0/(distances+1e-8)
#         inv_d = np.pow(inv_d, 10)
#         p_weight = inv_d / np.sum(inv_d)
#         idx = np.random.choice(len(distances), p=p_weight)
#         closest_coord = tuple(valid_coords[idx])
#         result[y-min_h, x-min_w] = closest_coord
    
    return [patch_coor_to_ind(x,y,w,txt_len) for y, x in result.flatten()]

def get_interpolated_patch_ind(h, w, bbox_coordinates, img_ids, patch_size=16, txt_len=512):
    # Get only valid coordinates (value == 1)
    large_array = np.ones((h,w), dtype=int)
    indices_array = img_ids.cpu().numpy().squeeze().copy()
    indices_array = indices_array.reshape((h,w,-1))

    small_grid_coords, bbox_h, bbox_w = bbox_to_patch_coords(bbox_coordinates, patch_size=patch_size, return_shape=True)

    for y, x in small_grid_coords:
        large_array[y,x] = 0
        indices_array[y,x,:] = 0
    # return None
    
    indices_array = indices_array.reshape((h*w,-1))
    indices_array = indices_array[np.argwhere(large_array.flatten()==1)].squeeze()

    valid_coords = np.argwhere(large_array == 1)
    min_h, min_w = small_grid_coords[0]
    
    result = list()
    for idx, coord in enumerate(small_grid_coords):
        y, x = coord
        # distances = np.abs(valid_coords[:, 0] - y) + np.abs(valid_coords[:, 1] - x)
        distances = np.sqrt((valid_coords[:, 0] - y)**2 + (valid_coords[:, 1] - x)**2)
        min_idx = np.argmin(distances)
        inv_d = 1.0/(distances+1e-8)
        inv_d = np.pow(inv_d, 10)
        p_weight = inv_d / np.sum(inv_d)
        interp_ind = (p_weight.reshape(-1,1) * indices_array).sum(0)
        # import pdb;pdb.set_trace()
        result.append(interp_ind)
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

def visualize_grid(large_array, small_grid_coords, closest_grid, filename="output_grid.png"):
    fig, ax = plt.subplots()
    ax.imshow(large_array, cmap='Greys', origin='lower', alpha=0.2)

    # Valid coordinates
    valid_coords = np.argwhere(large_array == 1)
    ax.plot(valid_coords[:, 1], valid_coords[:, 0], 'ro', markersize=3, label='Valid Coordinates')

    # Small grid
    small_coords = np.array(small_grid_coords)
    ax.plot(small_coords[:, 1], small_coords[:, 0], 'bo', markersize=3, label='Small Grid Coords')

    # Arrows
    h, w = closest_grid.shape
    for i in range(h):
        for j in range(w):
            src = small_grid_coords[i * w + j]
            dst = closest_grid[i, j]
            ax.arrow(src[1], src[0], dst[1] - src[1], dst[0] - src[0],
                     head_width=0.2, head_length=0.3, fc='gray', ec='gray', length_includes_head=True)

    ax.set_title("Closest Valid Coordinates (Excluding Small Grid)")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.grid(True)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to '{filename}'")


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_patch_assignments(h, w, bbox_coords, neighbor_lists, filename="patch_assignments.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, which='both', color='lightgrey', linestyle='--', linewidth=0.5)

    # Plot bbox patches in black
    for y, x in bbox_coords:
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1.5, edgecolor='black', facecolor='black')
        ax.add_patch(rect)

    # Assign a color to each bbox patch's neighbor group
    cmap = plt.cm.get_cmap('tab20', len(neighbor_lists))
    for idx, neighbors in enumerate(neighbor_lists):
        color = cmap(idx)
        for y, x in neighbors:
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, edgecolor=color, facecolor=color, alpha=0.7)
            ax.add_patch(rect)

    ax.set_xticks(np.arange(0, w+1, 1))
    ax.set_yticks(np.arange(0, h+1, 1))
    plt.title("Neighbors per BBox Patch")

    # Save to file
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def visualize_each_patch_assignment(h, w, bbox_coords, neighbor_lists, output_dir="patch_figures"):
    os.makedirs(output_dir, exist_ok=True)

    for idx, (center_patch, neighbors) in enumerate(zip(bbox_coords, neighbor_lists)):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, which='both', color='lightgrey', linestyle='--', linewidth=0.5)

        # Draw all bbox patches (in black)
        for y, x in bbox_coords:
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1.0, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        # Highlight the current bbox patch (in red)
        cy, cx = center_patch
        rect = patches.Rectangle((cx, cy), 1, 1, linewidth=2.0, edgecolor='red', facecolor='red')
        ax.add_patch(rect)

        # Draw its neighbors (in blue)
        for y, x in neighbors:
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, edgecolor='blue', facecolor='blue', alpha=0.7)
            ax.add_patch(rect)

        ax.set_xticks(np.arange(0, w + 1, 1))
        ax.set_yticks(np.arange(0, h + 1, 1))
        plt.title(f"Neighbors for BBox Patch #{idx} at ({cy}, {cx})")

        filename = os.path.join(output_dir, f"patch_{idx}_{cy}_{cx}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

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

def shape_coords_to_patch_indices(coords, h, w, patch_size=16, txt_len=512):
    """
    Convert a list of pixel coordinates to patch indices.
    
    Args:
        coords: List of (y, x) pixel coordinates
        h, w: Image height and width
        patch_size: Size of each patch
        txt_len: Text length offset
    
    Returns:
        List of unique patch indices
    """
    patch_h, patch_w = h // patch_size, w // patch_size
    patch_coords = set()
    
    for y, x in coords:
        # Convert pixel coordinates to patch coordinates
        py, px = y // patch_size, x // patch_size
        if 0 <= py < patch_h and 0 <= px < patch_w:
            patch_coords.add((py, px))
    
    # Convert to indices
    indices = [patch_coor_to_ind(px, py, patch_w, txt_len) for py, px in patch_coords]
    return indices

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

def get_shape_closest_patch_ind(h, w, shape_indices, patch_size=16, txt_len=512):
    """
    Get closest patch indices for arbitrary shape (replacement for removal artifacts).
    
    Args:
        h, w: Patch grid dimensions
        shape_indices: List of patch indices defining the shape
        patch_size: Size of each patch
        txt_len: Text length offset
    
    Returns:
        List of closest patch indices outside the shape
    """
    # Create a mask of occupied patches
    large_array = np.ones((h, w), dtype=int)
    shape_coords = [patch_ind_to_coor(ind, w, txt_len) for ind in shape_indices]
    
    for y, x in shape_coords:
        if 0 <= y < h and 0 <= x < w:
            large_array[y, x] = 0
    
    valid_coords = np.argwhere(large_array == 1)
    
    # Find closest valid patch for each shape patch
    result = []
    for y, x in shape_coords:
        if 0 <= y < h and 0 <= x < w:
            distances = np.abs(valid_coords[:, 0] - y) + np.abs(valid_coords[:, 1] - x)
            min_idx = np.argmin(distances)
            closest_coord = tuple(valid_coords[min_idx])
            result.append(patch_coor_to_ind(closest_coord[1], closest_coord[0], w, txt_len))
    
    return result

def get_shape_neighbors_patch_ind(h, w, shape_indices, img_ids, patch_size=16, txt_len=512):
    """
    Get neighbor patch features for arbitrary shape.
    
    Args:
        h, w: Patch grid dimensions
        shape_indices: List of patch indices defining the shape
        img_ids: Image ID tensor
        patch_size: Size of each patch
        txt_len: Text length offset
    
    Returns:
        Tensor of averaged neighbor features
    """
    shape_coords = np.array([patch_ind_to_coor(ind, w, txt_len) for ind in shape_indices])
    
    indices_array = img_ids.cpu().numpy().squeeze().copy()
    indices_array = indices_array.reshape((h*w, -1))
    
    # Compute shape center and radius
    min_y, min_x = np.min(shape_coords, axis=0)
    max_y, max_x = np.max(shape_coords, axis=0)
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2
    
    radius_y = min(center_y - min_y, max_y - center_y)
    radius_x = min(center_x - min_x, max_x - center_x)
    radius = min(radius_y, radius_x)
    
    # Create set of shape coordinates for fast lookup
    shape_set = set(map(tuple, shape_coords))
    
    result = []
    for center_y, center_x in shape_coords:
        neighbors = []
        for dy in range(-radius-1, radius + 2):
            for dx in range(-radius-1, radius + 2):
                if abs(dy) + abs(dx) <= radius + 1:
                    ny, nx = center_y + dy, center_x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if (ny, nx) not in shape_set:
                            neighbors.append((ny, nx))
        
        if neighbors:
            neighbors_ind = indices_array[[patch_coor_to_ind(x, y, w, 0) for (y, x) in neighbors], :]
            result.append(neighbors_ind.mean(0))
        else:
            # Fallback if no neighbors found
            result.append(np.zeros(indices_array.shape[1]))
    
    return torch.from_numpy(np.array(result)).unsqueeze(0)

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
            inv_d = np.pow(inv_d, 10)
            p_weight = inv_d / np.sum(inv_d)
            idx = np.random.choice(len(distances), p=p_weight)
            closest_coord = tuple(reference_coords[idx])
            result.append(patch_coor_to_ind(closest_coord[1], closest_coord[0], w, txt_len))
    
    return result

def get_shape_interpolated_patch_ind(h, w, shape_indices, img_ids, patch_size=16, txt_len=512):
    """
    Get interpolated patch features for arbitrary shape.
    
    Args:
        h, w: Patch grid dimensions
        shape_indices: List of patch indices defining the shape
        img_ids: Image ID tensor
        patch_size: Size of each patch
        txt_len: Text length offset
    
    Returns:
        Tensor of interpolated features
    """
    large_array = np.ones((h, w), dtype=int)
    indices_array = img_ids.cpu().numpy().squeeze().copy()
    indices_array = indices_array.reshape((h, w, -1))
    
    shape_coords = [patch_ind_to_coor(ind, w, txt_len) for ind in shape_indices]
    
    for y, x in shape_coords:
        if 0 <= y < h and 0 <= x < w:
            large_array[y, x] = 0
            indices_array[y, x, :] = 0
    
    indices_array = indices_array.reshape((h*w, -1))
    indices_array = indices_array[np.argwhere(large_array.flatten()==1)].squeeze()
    
    valid_coords = np.argwhere(large_array == 1)
    
    result = []
    for y, x in shape_coords:
        if 0 <= y < h and 0 <= x < w:
            distances = np.sqrt((valid_coords[:, 0] - y)**2 + (valid_coords[:, 1] - x)**2)
            inv_d = 1.0 / (distances + 1e-8)
            inv_d = np.pow(inv_d, 10)
            p_weight = inv_d / np.sum(inv_d)
            interp_ind = (p_weight.reshape(-1, 1) * indices_array).sum(0)
            result.append(interp_ind)
    
    return torch.from_numpy(np.array(result)).unsqueeze(0)

def perturb_shape_pe(h, w, shape_indices, img_ids, patch_size=16, txt_len=512):
    """
    Perturb positional encoding for arbitrary shape.
    
    Args:
        h, w: Patch grid dimensions
        shape_indices: List of patch indices defining the shape
        img_ids: Image ID tensor
        patch_size: Size of each patch
        txt_len: Text length offset
    
    Returns:
        Tensor of perturbed positional encodings
    """
    indices_array = torch.from_numpy(img_ids.cpu().numpy().squeeze().copy()[shape_indices, :])
    noise = torch.randn_like(indices_array)
    
    # Mask for non-zero elements
    nonzero_mask = indices_array != 0
    perturbed_indices_array = indices_array.clone()
    perturbed_indices_array[nonzero_mask] += noise[nonzero_mask]
    
    return perturbed_indices_array.unsqueeze(0)

# def shuffle_pe(h, w, shape_indices, patch_size=16, txt_len=512, intensity=3):
#     """
#     Shuffle positional encoding for arbitrary shape.
    
#     Args:
#         h, w: Patch grid dimensions
#         shape_indices: List of patch indices defining the shape
#         patch_size: Size of each patch
#         txt_len: Text length offset
#         intensity: Shuffle intensity
    
#     Returns:
#         List of shuffled patch indices
#     """
#     shape_coords = [patch_ind_to_coor(ind, w, txt_len) for ind in shape_indices]
    
#     shuffled_coords = []
#     for y, x in shape_coords:
#         dx = torch.randint(-intensity, intensity + 1, (1,)).item()
#         dy = torch.randint(-intensity, intensity + 1, (1,)).item()
        
#         new_x = max(0, min(x + dx, w - 1))
#         new_y = max(0, min(y + dy, h - 1))
        
#         shuffled_coords.append((new_y, new_x))
    
#     return [patch_coor_to_ind(x, y, w, txt_len) for y, x in shuffled_coords]

def visualize_mask_patches(mask, patch_size=16, txt_len=512, filename="mask_patches_visualization.png", show_grid=True):
    """
    Visualize the patches retrieved from a mask using mask_to_patch_indices.
    
    Args:
        mask: Binary mask (numpy array) where True indicates the region of interest
        patch_size: Size of each patch (default 16 for Flux)
        txt_len: Text length offset (default 512)
        filename: Output filename for the visualization
        show_grid: Whether to show the patch grid overlay
    
    Returns:
        tuple: (patch_indices, patch_coordinates, patch_mask_grid)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    h, w = mask.shape
    patch_h, patch_w = h // patch_size, w // patch_size
    
    # Get patch indices using the function
    patch_indices = mask_to_patch_indices(mask, patch_size, txt_len)
    
    # Convert indices back to coordinates for visualization
    patch_coords = patch_indices_to_coords(patch_indices, patch_w, txt_len)
    
    # Create patch-level mask grid
    patch_mask_grid = np.zeros((patch_h, patch_w), dtype=bool)
    for py, px in patch_coords:
        if 0 <= py < patch_h and 0 <= px < patch_w:
            patch_mask_grid[py, px] = True
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original mask
    axes[0].imshow(mask, cmap='gray', origin='upper')
    axes[0].set_title(f'Original Mask\n({h}×{w} pixels)')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    if show_grid:
        # Add patch grid overlay
        for i in range(0, h, patch_size):
            axes[0].axhline(y=i, color='red', alpha=0.3, linewidth=0.5)
        for j in range(0, w, patch_size):
            axes[0].axvline(x=j, color='red', alpha=0.3, linewidth=0.5)
    
    # Plot 2: Patch-level mask
    axes[1].imshow(patch_mask_grid, cmap='gray', origin='upper')
    axes[1].set_title(f'Patch-Level Mask\n({patch_h}×{patch_w} patches)')
    axes[1].set_xlabel('X (patches)')
    axes[1].set_ylabel('Y (patches)')
    
    # Add grid for patch visualization
    for i in range(patch_h + 1):
        axes[1].axhline(y=i-0.5, color='blue', alpha=0.5, linewidth=0.5)
    for j in range(patch_w + 1):
        axes[1].axvline(x=j-0.5, color='blue', alpha=0.5, linewidth=0.5)
    
    # Plot 3: Overlay showing selected patches on original image
    axes[2].imshow(mask, cmap='gray', alpha=0.7, origin='upper')
    
    # Overlay selected patches
    for py, px in patch_coords:
        if 0 <= py < patch_h and 0 <= px < patch_w:
            # Convert patch coordinates back to pixel coordinates
            y_start, y_end = py * patch_size, (py + 1) * patch_size
            x_start, x_end = px * patch_size, (px + 1) * patch_size
            
            # Draw patch boundary
            rect = patches.Rectangle(
                (x_start, y_start), patch_size, patch_size,
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
            )
            axes[2].add_patch(rect)
    
    axes[2].set_title(f'Selected Patches Overlay\n({len(patch_indices)} patches selected)')
    axes[2].set_xlabel('X (pixels)')
    axes[2].set_ylabel('Y (pixels)')
    
    # Add patch grid overlay
    if show_grid:
        for i in range(0, h, patch_size):
            axes[2].axhline(y=i, color='blue', alpha=0.3, linewidth=0.5)
        for j in range(0, w, patch_size):
            axes[2].axvline(x=j, color='blue', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mask visualization saved to '{filename}'")
    print(f"Original mask: {h}×{w} pixels")
    print(f"Patch grid: {patch_h}×{patch_w} patches (patch_size={patch_size})")
    print(f"Selected patches: {len(patch_indices)} out of {patch_h * patch_w} total patches")
    print(f"Patch indices (first 10): {patch_indices[:10]}{'...' if len(patch_indices) > 10 else ''}")
    print(f"Patch coordinates (first 10): {patch_coords[:10]}{'...' if len(patch_coords) > 10 else ''}")
    
    return patch_indices, patch_coords, patch_mask_grid

def compare_mask_methods(mask, patch_size=16, txt_len=512, filename="mask_methods_comparison.png"):
    """
    Compare different ways of representing the same mask for patch-based processing.
    
    Args:
        mask: Binary mask (numpy array)
        patch_size: Size of each patch
        txt_len: Text length offset
        filename: Output filename
    
    Returns:
        dict: Dictionary containing results from different methods
    """
    import matplotlib.pyplot as plt
    from shape_utils import ShapeProcessor
    
    h, w = mask.shape
    patch_h, patch_w = h // patch_size, w // patch_size
    
    # Method 1: mask_to_patch_indices
    indices_from_mask = mask_to_patch_indices(mask, patch_size, txt_len)
    
    # Method 2: shape_coords_to_patch_indices (convert mask to coordinates first)
    coords = ShapeProcessor.mask_to_coords(mask, sample_rate=1)
    indices_from_coords = shape_coords_to_patch_indices(coords, h, w, patch_size, txt_len)
    
    # Method 3: downsampled mask approach (for comparison)
    downsampled_mask = np.zeros((patch_h, patch_w), dtype=bool)
    for py in range(patch_h):
        for px in range(patch_w):
            y_start, y_end = py * patch_size, (py + 1) * patch_size
            x_start, x_end = px * patch_size, (px + 1) * patch_size
            patch_region = mask[y_start:y_end, x_start:x_end]
            downsampled_mask[py, px] = np.any(patch_region)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Original mask and coordinate representation
    axes[0, 0].imshow(mask, cmap='gray', origin='upper')
    axes[0, 0].set_title(f'Original Mask\n{np.sum(mask)} pixels')
    
    # Show sampled coordinates
    if len(coords) > 0:
        y_coords, x_coords = zip(*coords)
        axes[0, 1].imshow(mask, cmap='gray', alpha=0.3, origin='upper')
        axes[0, 1].scatter(x_coords[::max(1, len(coords)//1000)], 
                          y_coords[::max(1, len(coords)//1000)], 
                          c='red', s=1, alpha=0.7)
        axes[0, 1].set_title(f'Coordinate Representation\n{len(coords)} coordinates')
    else:
        axes[0, 1].text(0.5, 0.5, 'No coordinates found', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Coordinate Representation\n0 coordinates')
    
    # Downsampled mask
    axes[0, 2].imshow(downsampled_mask, cmap='gray', origin='upper')
    axes[0, 2].set_title(f'Downsampled Mask\n{patch_h}×{patch_w} patches')
    
    # Bottom row: Patch representations
    patch_coords_1 = patch_indices_to_coords(indices_from_mask, patch_w, txt_len)
    patch_mask_1 = np.zeros((patch_h, patch_w), dtype=bool)
    for py, px in patch_coords_1:
        if 0 <= py < patch_h and 0 <= px < patch_w:
            patch_mask_1[py, px] = True
    
    axes[1, 0].imshow(patch_mask_1, cmap='gray', origin='upper')
    axes[1, 0].set_title(f'From mask_to_patch_indices\n{len(indices_from_mask)} patches')
    
    patch_coords_2 = patch_indices_to_coords(indices_from_coords, patch_w, txt_len)
    patch_mask_2 = np.zeros((patch_h, patch_w), dtype=bool)
    for py, px in patch_coords_2:
        if 0 <= py < patch_h and 0 <= px < patch_w:
            patch_mask_2[py, px] = True
    
    axes[1, 1].imshow(patch_mask_2, cmap='gray', origin='upper')
    axes[1, 1].set_title(f'From shape_coords_to_patch_indices\n{len(indices_from_coords)} patches')
    
    # Difference between methods
    diff_mask = patch_mask_1.astype(int) - patch_mask_2.astype(int)
    axes[1, 2].imshow(diff_mask, cmap='RdBu', vmin=-1, vmax=1, origin='upper')
    axes[1, 2].set_title(f'Difference (Method 1 - Method 2)\n{np.sum(np.abs(diff_mask))} different patches')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'indices_from_mask': indices_from_mask,
        'indices_from_coords': indices_from_coords,
        'patch_coords_from_mask': patch_coords_1,
        'patch_coords_from_coords': patch_coords_2,
        'downsampled_mask': downsampled_mask,
        'methods_match': len(set(indices_from_mask) - set(indices_from_coords)) == 0
    }
    
    print(f"Comparison saved to '{filename}'")
    print(f"Method 1 (mask_to_patch_indices): {len(indices_from_mask)} patches")
    print(f"Method 2 (shape_coords_to_patch_indices): {len(indices_from_coords)} patches")
    print(f"Methods produce identical results: {results['methods_match']}")
    
    return results