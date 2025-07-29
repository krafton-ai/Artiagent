import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str], 
           info=None) -> dict[str, Tensor]:
    """
    Prepare inputs for the flux model with support for patch indices.
    
    Args:
        t5, clip: Text encoders
        img: Input image tensor
        prompt: Text prompt(s)
        info: Additional information dictionary, must contain 'artifact_data'.
    
    Returns:
        Dictionary containing prepared inputs
    """
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    patch_h, patch_w = h // 2, w // 2
    
    # Add patch dimensions to info for model to use
    info['patch_h'] = patch_h
    info['patch_w'] = patch_w

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }, (patch_h, patch_w)


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def denoise_first_order(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info,
    percentage_of_steps = 1.0,
    guidance: float = 5.0
):
    # this is ignored for schnell
    inject_list = [True] * int(info['inject_step']) + [False] * (int(len(timesteps) * percentage_of_steps) -1 - int(info['inject_step']))
    attn_mask_list = [True] * int(info['attn_mask_step']) + [False] * (int(len(timesteps) * percentage_of_steps) -1 - int(info['attn_mask_step']))
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    if percentage_of_steps != 1:
        end_timestep_idx = int(len(timesteps) * percentage_of_steps)
        if inverse:
            timesteps = timesteps[:end_timestep_idx]
            # inject_list = inject_list[:end_timestep_idx - 1]
        else:
            timesteps = timesteps[len(timesteps) - end_timestep_idx:]
            # inject_list = inject_list[len(inject_list) - end_timestep_idx + 1:]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]
        info['attn_mask'] = attn_mask_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img = img + (t_prev - t_curr) * pred
    return img, info


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    percentage_of_steps = 1.0,
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * int(info['inject_step']) + [False] * (int(len(timesteps) * percentage_of_steps) -1 - int(info['inject_step']))
    attn_mask_list = [True] * int(info['attn_mask_step']) + [False] * (int(len(timesteps) * percentage_of_steps) -1 - int(info['attn_mask_step']))

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        # patch_ids = None
        # info['double_stream_block_mask'] = 'none'
        # info['single_stream_block_mask'] = 'none'

    if percentage_of_steps != 1:
        end_timestep_idx = int(len(timesteps) * percentage_of_steps)
        if inverse:
            timesteps = timesteps[:end_timestep_idx]
            # inject_list = inject_list[:end_timestep_idx - 1]
        else:
            timesteps = timesteps[len(timesteps) - end_timestep_idx:]
            # inject_list = inject_list[len(inject_list) - end_timestep_idx + 1:]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]
        info['attn_mask'] = attn_mask_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
