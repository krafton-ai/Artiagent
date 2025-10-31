import torch
import argparse
from typing import Dict, Optional, Union, List
import numpy as np
from dataclasses import dataclass
import os
import re
import json
import time
from glob import iglob
from einops import rearrange
from PIL import Image

# FLUX imports
import flux
from flux.sampling import denoise, denoise_first_order, denoise_fireflow, get_schedule, prepare, unpack
from flux.util import (configs, load_ae, load_clip,
                        load_flow_model, load_t5)
from flux.math import get_attn_mask

@dataclass
class FluxConfig:
    """Configuration for FLUX model"""
    name: str = 'flux-dev'
    guidance: float = 5.0
    num_steps: int = 25
    inject_step: int = 15
    pe_step: Union[int, Dict[str, int]] = 25  # Can be int or dict with artifact type keys
    attn_mask_step: int = 0
    seed: int = 42
    masks: list = None
    alpha: float = 0.0
    feature_path: str = 'feature'
    percentage_of_steps: float = 1.0
    offload: bool = False
    use_rf_solver: bool = False  # Use denoise (RF solver) instead of denoise_first_order
    
    def __post_init__(self):
        if self.masks is None:
            self.masks = ['none', 'none', 'none', 'none']
        
        # Validate pe_step configuration
        if isinstance(self.pe_step, dict):
            required_keys = {'addition', 'removal', 'distortion', 'fusion'}
            provided_keys = set(self.pe_step.keys())
            if not required_keys.issubset(provided_keys):
                missing_keys = required_keys - provided_keys
                raise ValueError(f"pe_step dict missing required artifact types: {missing_keys}")
            
            # Validate all values are integers
            for artifact_type, value in self.pe_step.items():
                if not isinstance(value, int):
                    raise ValueError(f"pe_step value for '{artifact_type}' must be an integer, got {type(value)}")
    
    def get_pe_step(self, artifact_type: str) -> int:
        """
        Get pe_step value for specific artifact type
        
        Args:
            artifact_type: Type of artifact ('addition', 'removal', 'distortion')
            
        Returns:
            pe_step value for the artifact type
        """
        if isinstance(self.pe_step, dict):
            if artifact_type not in self.pe_step:
                raise ValueError(f"Unknown artifact type '{artifact_type}'. Available types: {list(self.pe_step.keys())}")
            return self.pe_step[artifact_type]
        else:
            return self.pe_step


class FluxGenerator:
    """Handler for FLUX model operations and image generation"""
    
    def __init__(self, device: str = 'cuda', config: Optional[FluxConfig] = None):
        """
        Initialize FLUX generator
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            config: FLUX configuration object
        """
        self.device = device
        self.config = config or FluxConfig()
        
        # Model components
        self.t5 = None
        self.clip = None
        self.model = None
        self.ae = None
        
        self._models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all FLUX model components"""
        print("Loading FLUX models...")
        
        # Determine max_length based on model name
        max_length = 256 if self.config.name == "flux-schnell" else 512
        
        # Load model components
        self.t5 = load_t5(self.device, max_length=max_length)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.config.name, device=self.device)
        self.ae = load_ae(self.config.name, device=self.device)
        
        self._models_loaded = True
        print("FLUX models loaded successfully.")
    
    def create_default_flux_args(self) -> argparse.Namespace:
        """
        Create default FLUX arguments based on current configuration
        
        Returns:
            Default argparse.Namespace object with config values
        """
        # Create parser and args
        parser = argparse.ArgumentParser()
        flux_args = parser.parse_args(args=[])
        
        # Set FLUX configuration defaults
        flux_args.name = self.config.name
        flux_args.feature_path = self.config.feature_path
        flux_args.guidance = self.config.guidance
        flux_args.num_steps = self.config.num_steps
        flux_args.inject_step = self.config.inject_step
        flux_args.attn_mask_step = self.config.attn_mask_step
        flux_args.pe_step = self.config.pe_step
        flux_args.pe_step_addition = self.config.pe_step['addition']
        flux_args.pe_step_removal = self.config.pe_step['removal']
        flux_args.pe_step_distortion = self.config.pe_step['distortion']
        flux_args.pe_step_fusion = self.config.pe_step['fusion']
        flux_args.seed = self.config.seed
        flux_args.masks = self.config.masks.copy()
        flux_args.alpha = self.config.alpha
        flux_args.percentage_of_steps = self.config.percentage_of_steps
        flux_args.offload = self.config.offload
        
        # Initialize task-specific arguments to None
        flux_args.source_prompt = None
        flux_args.target_prompt = None
        flux_args.artifact_type = None
        flux_args.output_dir = None
        flux_args.source_img = None
        
        # Initialize optional patch mapping information
        flux_args.patch_mapping = None
        flux_args.reference_patch_indices = None
        flux_args.target_patch_indices = None
        
        return flux_args

    @torch.inference_mode()
    def inject_artifacts(self,
        source_prompt: str,
        target_prompt: str,
        artifact_data: Dict,
        source_img: Union[np.ndarray, str],
        output_dir: str = None,
        pe_step_addition: Optional[int] = None,
        pe_step_removal: Optional[int] = None,
        pe_step_distortion: Optional[int] = None,
        pe_step_fusion: Optional[int] = None,
        inject_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        use_fireflow: bool = False,
    ):
        """
        Sample the flux model with artifact injection supporting arbitrary shapes.
        
        Args:
            source_prompt: Source image prompt/caption
            target_prompt: Target prompt for generation
            artifact_type: Type of artifact ('addition', 'removal', 'distortion')
            source_img: Source image array or path
            output_dir: Output directory for generated images
            reference_patch_indices: List of reference patch indices
            target_patch_indices: List of target patch indices
        """
        torch.set_grad_enabled(False)

        # Create default flux args and update with passed parameters
        flux_args = self.create_default_flux_args()
        
        # Update with required parameters
        flux_args.source_prompt = source_prompt
        flux_args.target_prompt = target_prompt
        flux_args.artifact_data = artifact_data
        flux_args.source_img = source_img
        flux_args.output_dir = output_dir
        flux_args.inject_step = inject_step if inject_step is not None else self.config.inject_step
        torch_device = torch.device(self.device)
        if num_steps is not None:
            flux_args.num_steps = num_steps
            
        init_image = None
        init_image = self.load_image(flux_args.source_img)
        
        shape = init_image.shape

        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        init_image = init_image[:new_h, :new_w, :]

        width, height = init_image.shape[0], init_image.shape[1]
        init_image = self.encode(init_image, torch_device, self.ae)

        rng = torch.Generator(device="cpu").manual_seed(flux_args.seed)

        if flux_args.seed is None:
            flux_args.seed = rng.seed()
        print(f"Generating with seed {flux_args.seed}:\n{flux_args.source_prompt}")
        t0 = time.perf_counter()

        flux_args.seed = None
        if flux_args.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)

        info = {}
        info['feature_path'] = flux_args.feature_path
        info['feature'] = {}
        info['inject_step'] = flux_args.inject_step
        info['attn_mask_step'] = flux_args.attn_mask_step
        info['alpha'] = flux_args.alpha
        if pe_step_distortion is not None:
            info['pe_step_distortion'] = pe_step_distortion
        else:
            info['pe_step_distortion'] = flux_args.pe_step_distortion
        if pe_step_removal is not None:
            info['pe_step_removal'] = pe_step_removal
        else:
            info['pe_step_removal'] = flux_args.pe_step_removal
        if pe_step_addition is not None:
            info['pe_step_addition'] = pe_step_addition
        else:
            info['pe_step_addition'] = flux_args.pe_step_addition
        if pe_step_fusion is not None:
            info['pe_step_fusion'] = pe_step_fusion
        else:
            info['pe_step_fusion'] = flux_args.pe_step_fusion
        info['artifact_data'] = flux_args.artifact_data
        info['guidance'] = flux_args.guidance
        if not os.path.exists(flux_args.feature_path):
            os.mkdir(flux_args.feature_path)

        # Prepare inputs with shape-aware approach
        inp, (patch_h, patch_w) = prepare(
            self.t5, self.clip, init_image, 
            prompt=flux_args.source_prompt,
            info=info
        )
        
        inp_target, _ = prepare(
            self.t5, self.clip, init_image, 
            prompt=flux_args.target_prompt,
            info=info
        )
        
        timesteps = get_schedule(flux_args.num_steps, inp["img"].shape[1], shift=(flux_args.name != "flux-schnell"))

        info['patch_h'] = patch_h
        info['patch_w'] = patch_w

        L = inp['img'].shape[1] + inp['txt'].shape[1]
        
        # Choose denoising function based on configuration
        # RF solver (denoise) is more accurate but slower than first-order denoising
        denoise_func = denoise_fireflow if use_fireflow else denoise_first_order
        # denoise_func = denoise_fireflow
        # denoise_func = denoise_fireflow
        # inversion initial noise
        z, info = denoise_func(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info, percentage_of_steps=flux_args.percentage_of_steps)
        inp_target["img"] = z

        timesteps = get_schedule(flux_args.num_steps, inp_target["img"].shape[1], shift=(flux_args.name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise_func(self.model, **inp_target, timesteps=timesteps, guidance=info['guidance'], inverse=False, info=info, percentage_of_steps=flux_args.percentage_of_steps)

        # decode latents to pixel space

        #######################################
        #### TODO: allow batch computation ####
        #######################################

        x = unpack(x.float(), width, height)

        if output_dir is not None:
            output_name = os.path.join(output_dir, "img.jpg")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                idx = 0
            else:
                fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
                if len(fns) > 0:
                    idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                else:
                    idx = 0

        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")

        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img
    
    def load_image(self, source):
        """Load image from various sources (numpy array, PIL Image, or file path)"""
        if isinstance(source, np.ndarray):
            # Already a NumPy array
            return source
        elif isinstance(source, Image.Image):
            # Already a PIL Image
            return np.array(source.convert('RGB'))
        elif isinstance(source, str):
            if os.path.isfile(source):
                # It's a file path to an image
                return np.array(Image.open(source).convert('RGB'))
            else:
                raise ValueError(f"Provided string is not a valid file: {source}")
        else:
            raise TypeError(f"Unsupported input type: {type(source)}")
    
    @torch.inference_mode()
    def encode(self, init_image, torch_device, ae):
        """Encode image to latent space"""
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
        return init_image
    
    def create_attention_mask(self, mask_type: str, L: int, patch_ids: list, txt_len: int, device, dtype) -> torch.Tensor:
        """
        Pre-compute attention mask to avoid repeated computation during forward pass.
        
        Args:
            mask_type: Type of attention mask ('none', 'm1', 'no_text', etc.)
            L: Sequence length
            patch_ids: List of patch indices
            txt_len: Text sequence length
            device: Device to create tensor on
            dtype: Data type for tensor
        
        Returns:
            Pre-computed attention mask tensor
        """
        v_ids = set(range(512, L))
        v_ids = list(v_ids-set(patch_ids)-set(range(0,512)))
        attn_map = torch.zeros(L, L, dtype=dtype, device=device)
        not_patch_ids = list(set(range(0,L))-set(patch_ids))
        not_patch_ids_tensor = torch.tensor(not_patch_ids, device=device)
        not_v_ids = list(set(range(0,L))-set(v_ids))
        not_v_ids_tensor = torch.tensor(not_v_ids, device=device)
        v_ids_tensor = torch.tensor(v_ids, device=device)
        patch_ids_tensor = torch.tensor(patch_ids, device=device)
        
        if mask_type == 'none':
            return attn_map
        if mask_type == 'm1':
            attn_map[:txt_len, txt_len:512] = float('-inf')
            attn_map[txt_len:512, :txt_len] = float('-inf')
            attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
            attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
        elif mask_type == 'no_text':
            attn_map[:512, :] = float('-inf')
            attn_map[:, :512] = float('-inf')
            attn_map[:txt_len, :txt_len] = 0.0
            attn_map[txt_len:512, txt_len:512] = 0.0
        elif mask_type == 'tv':
            attn_map[:512, v_ids_tensor] = float('-inf')
        elif mask_type == 'va':
            attn_map[v_ids_tensor.unsqueeze(0), patch_ids_tensor.unsqueeze(1)] = float('-inf')
            attn_map[patch_ids_tensor.unsqueeze(0), v_ids_tensor.unsqueeze(1)] = float('-inf')
        elif mask_type == 'txt_mask':
            attn_map[:txt_len, txt_len:512] = float('-inf')
            attn_map[txt_len:512, :txt_len] = float('-inf')
        elif mask_type == 'm4':
            attn_map[:txt_len, txt_len:512] = float('-inf')
            attn_map[txt_len:512, :txt_len] = float('-inf')
            attn_map[txt_len:512, patch_ids] = float('-inf')
            attn_map[patch_ids, txt_len:512] = float('-inf')
            attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
        elif mask_type == 'm5':
            attn_map[:,:] = float('-inf')
            attn_map[:txt_len, :txt_len] = 0.0
            attn_map[txt_len:512, txt_len:512] = 0.0
            attn_map[v_ids, :][:, v_ids] = 0.0
            attn_map[patch_ids, :][:, patch_ids] = 0.0
        elif mask_type == 'm6':
            attn_map[:512, 512:] = float('-inf')
            attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
            attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
        elif mask_type == 'm7':
            attn_map[1, 1:] = float('-inf')
            attn_map[1:512, patch_ids] = float('-inf')
            attn_map[v_ids_tensor, patch_ids_tensor] = float('-inf')
            attn_map[patch_ids, 1:512] = float('-inf')
            attn_map[patch_ids_tensor, v_ids_tensor] = float('-inf')
        elif mask_type == 'm8':
            attn_map[patch_ids_tensor.unsqueeze(0), v_ids_tensor.unsqueeze(1)] = float('-inf')
        elif mask_type == 'm9':
            attn_map[not_patch_ids_tensor.unsqueeze(0), patch_ids_tensor.unsqueeze(1)] = float('-inf')
            attn_map[patch_ids_tensor.unsqueeze(0), not_patch_ids_tensor.unsqueeze(1)] = float('-inf')
        else:
            raise ValueError(f"attention map {mask_type} does not exist")
        
        return attn_map

    
    def update_config(self, **kwargs):
        """Update FLUX configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def unload_models(self):
        """Unload models to free memory"""
        self.t5 = None
        self.clip = None
        self.model = None
        self.ae = None
        self._models_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.unload_models() 