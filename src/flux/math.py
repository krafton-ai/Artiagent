import math
import torch
from einops import rearrange
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

def get_attn_mask(m: str, query: Tensor, patch_ids: list, txt_len: int) -> Tensor:
    L = query.size(-2)
    v_ids = set(range(512, L))
    v_ids = list(v_ids-set(patch_ids)-set(range(0,512)))
    attn_map = torch.zeros(L, L, dtype=query.dtype, device=query.device)
    not_patch_ids = list(set(range(0,L))-set(patch_ids))
    not_patch_ids_tensor = torch.tensor(not_patch_ids)
    not_v_ids = list(set(range(0,L))-set(v_ids))
    not_v_ids_tensor = torch.tensor(not_v_ids)
    v_ids_tensor = torch.tensor(v_ids)
    patch_ids_tensor = torch.tensor(patch_ids)
    if m == 'none':
        return attn_map
    if m == 'm1':
        attn_map[:txt_len, txt_len:512] = float('-inf')
        attn_map[txt_len:512, :txt_len] = float('-inf')
        attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
    elif m == 'no_text':
        # attn_map[:512, 512:] = float('-inf')
        attn_map[:512, :] = float('-inf')
        attn_map[:, :512] = float('-inf')
        attn_map[:txt_len, :txt_len] = 0.0
        attn_map[txt_len:512, txt_len:512] = 0.0
        # attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        # attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
        # attn_map[patch_ids, :txt_len] = float('-inf')
    elif m == 'va':
        attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
    elif m == 'txt_mask':
        attn_map[:txt_len, txt_len:512] = float('-inf')
        attn_map[txt_len:512, :txt_len] = float('-inf')
        # return attn_map
    elif m == 'm4':
        attn_map[:txt_len, txt_len:512] = float('-inf')
        attn_map[txt_len:512, :txt_len] = float('-inf')
        attn_map[txt_len:512, patch_ids] = float('-inf')
        attn_map[patch_ids, txt_len:512] = float('-inf')
        attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        # attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
        # attn_map[:txt_len, :txt_len] = float('-inf')
    elif m == 'm5':
        attn_map[:,:] = float('-inf')
        attn_map[:txt_len, :txt_len] = 0.0
        attn_map[txt_len:512, txt_len:512] = 0.0
        attn_map[v_ids, :][:, v_ids] = 0.0
        attn_map[patch_ids, :][:, patch_ids] = 0.0
    elif m=='m6':
        attn_map[:512, 512:] = float('-inf')
        attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
    elif m=='m7':
        attn_map[1, 1:] = float('-inf')
        attn_map[1:512, patch_ids] = float('-inf')
        attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        attn_map[patch_ids, 1:512] = float('-inf')
        attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
    elif m == 'm8':
        # attn_map[:txt_len, txt_len:512] = float('-inf')
        # attn_map[:txt_len, v_ids] = float('-inf')
        # attn_map[txt_len:512, :txt_len] = float('-inf')
        # attn_map[txt_len:512, patch_ids] = float('-inf')
        # attn_map[v_ids, :txt_len] = float('-inf')
        # attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        # attn_map[patch_ids, txt_len:512] = float('-inf')
        attn_map[patch_ids_tensor.unsqueeze(0), v_ids_tensor.unsqueeze(1)] = float('-inf')
        # Convert to NumPy
        # grid_np = attn_map.float().cpu().numpy()

        # # Replace -inf with 0, and 0 with 1 for visualization
        # vis_grid = np.where(np.isneginf(grid_np), 0, 1)

        # # Plot
        # plt.figure(figsize=(5, 5))
        # plt.imshow(vis_grid, cmap='gray', interpolation='nearest')
        # plt.title('0 = White, -inf = Black')
        # plt.axis('off')  # Hide axis

        # # Save the image
        # plt.savefig('grid_visualization.png', bbox_inches='tight')
        # plt.show()
    elif m == 'm9':
        attn_map[not_patch_ids_tensor.unsqueeze(0), patch_ids_tensor.unsqueeze(1)] = float('-inf')
        attn_map[patch_ids_tensor.unsqueeze(0), not_patch_ids_tensor.unsqueeze(1)] = float('-inf')
    else:
        raise ValueError(f"attention map {m} does not exist")
    return attn_map


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_manipulated(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, patch_ids: list[int], txt_ids:list[int], alpha:float) -> Tensor:
    q, k = apply_rope(q, k, pe)
    
    x = scaled_dot_product_attention_manipulated(q, k, v, patch_ids, txt_ids, alpha)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_manipulated_masked(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, patch_ids: list[int], txt_ids:list[int], mask: Tensor, alpha:float) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # x = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)
    x = scaled_dot_product_attention_manipulated_masked(q, k, v, patch_ids, txt_ids, mask, alpha)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_masked(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, patch_ids: list[int], mask: Tensor, return_weight:bool=False) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # x = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)
    if return_weight:
        x, m = scaled_dot_product_attention_masked(q, k, v, patch_ids, mask, return_weight)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x, m
    else:
        x = scaled_dot_product_attention_masked(q, k, v, patch_ids, mask, return_weight)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x

    # return x, m

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def scaled_dot_product_attention_manipulated_masked(query, key, value, patch_ids, txt_ids, attn_mask, alpha):
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    uniform_target = torch.zeros_like(attn_weight[:, :, patch_ids, :3])
    uniform_target[:,:,:,:2] = attn_weight[:,:,patch_ids, :3].sum(3).unsqueeze(3) / len(txt_ids)
    attn_weight[:, :, patch_ids, :3] = (
        (1 - alpha) * attn_weight[:, :, patch_ids, :3] +
        alpha * uniform_target
    )
    attn_weight[:, :, patch_ids, :] = attn_weight[:, :, patch_ids, :] / attn_weight[:, :, patch_ids, :].sum(dim=-1, keepdim=True)
    return attn_weight @ value

def scaled_dot_product_attention_masked(query, key, value, patch_ids, attn_mask, return_weight=False):
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if return_weight:
        return attn_weight @ value, attn_weight
    return attn_weight @ value

def scaled_dot_product_attention_manipulated(query, key, value, patch_ids, txt_ids, alpha=1.0, return_weight=False):
    # text image 
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_mask = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # --- Vectorized interpolation ---
    # Create uniform target distribution over txt_ids
    # uniform_target = torch.zeros_like(attn_weight)
    uniform_target = torch.zeros_like(attn_weight[:, :, patch_ids, :512])
    uniform_target[:,:,:,txt_ids] = attn_weight[:,:,patch_ids, :3].sum(3).unsqueeze(3) / len(txt_ids)
    # uniform_target[:,:,:,txt_ids] = 10 / len(txt_ids)
    # Apply interpolation: attn_weight[patch_ids, txt_ids] = blend
    attn_weight[:, :, patch_ids, :512] = (
        (1 - alpha) * attn_weight[:, :, patch_ids, :512] +
        alpha * uniform_target[:, :, :, :512]
    )
    # attn_weight[:, :, patch_ids, :] = attn_weight[:, :, patch_ids, :] / attn_weight[:, :, patch_ids, :].sum(dim=-1, keepdim=True)
    return attn_weight @ value