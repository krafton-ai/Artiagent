from dataclasses import dataclass

import torch
from torch import Tensor, nn
import numpy as np

from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


from .artifacts_util import (
    shuffle_pe, sample_closest_patch_ind,  # Legacy functions
)


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        patch_ids = None,
        patch_ref_ids = None,
        guidance: Tensor | None = None,
        info = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        inject_pe = pe.clone()
        
        # Handle positional encoding manipulation for artifacts
        if (not info['inverse']) and (timesteps > info['pe_step']):
            if info['artifact_type'] == 'addition':
                inject_pe[:,:,patch_ids,:,:,:] = inject_pe[:,:,patch_ref_ids,:,:,:]
            elif info['artifact_type'] == 'removal':
                # Use shape-based approach for removal
                target_patch_ref_ids = sample_closest_patch_ind(
                    info['patch_h'], info['patch_w'], patch_ids, patch_ref_ids,
                    patch_size=16, txt_len=512
                )
                inject_pe[:,:,patch_ids,:,:,:] = inject_pe[:,:,target_patch_ref_ids,:,:,:]
                
            elif info['artifact_type'] == 'distortion':

                # Use shape-based approach for distortion
                # target_patch_ref_ids = shuffle_pe(
                #     info['patch_h'], info['patch_w'], patch_ids,
                #     patch_size=16, txt_len=512
                # )
                if len(patch_ref_ids) == 0:
                    target_patch_ref_ids = patch_ids.copy()
                    np.random.shuffle(target_patch_ref_ids)
                    inject_pe[:,:,patch_ids,:,:,:] = inject_pe[:,:,target_patch_ref_ids,:,:,:]
                else:
                    inject_pe[:,:,patch_ids,:,:,:] = inject_pe[:,:,patch_ref_ids,:,:,:]
            else:
                raise AttributeError('Artifact type is either not defined or unidentified')

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, info=info, patch_ids=patch_ids)

        cnt = 0
        img = torch.cat((txt, img), 1) 
        info['type'] = 'single'
        for block in self.single_blocks:
            info['id'] = cnt
            if cnt < 19:
                img, info = block(img, vec=vec, pe=inject_pe, info=info, patch_ids=patch_ids)
            else:
                img, info = block(img, vec=vec, pe=pe, info=info, patch_ids=patch_ids)
            cnt += 1
            

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img, info