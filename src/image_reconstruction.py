"""
Image Reconstruction Script

This script processes directories containing real_image.png files and reconstructs
them using FLUX model with two sequential denoise_fireflow calls.

The reconstruction process:
1. First denoise_fireflow with inverse=True (inversion to latent space)
2. Second denoise_fireflow with inverse=False (reconstruction from latent)

Usage:
    python image_reconstruction.py /path/to/parent/directory
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from einops import rearrange
import multiprocessing as mp

# FLUX imports
from flux.sampling import denoise_fireflow, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5


class ImageReconstructor:
    """Handler for FLUX-based image reconstruction"""
    
    def __init__(self, device: str = 'cuda', model_name: str = 'flux-dev', 
                 num_steps: int = 25, guidance: float = 1.0):
        """
        Initialize image reconstructor
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            model_name: FLUX model name ('flux-dev' or 'flux-schnell')
            num_steps: Number of denoising steps
            guidance: Guidance scale for generation
        """
        self.device = device
        self.model_name = model_name
        self.num_steps = num_steps
        self.guidance = guidance
        
        # Model components
        self.t5 = None
        self.clip = None
        self.model = None
        self.ae = None
        
        self.load_models()
    
    def load_models(self):
        """Load all FLUX model components"""
        print("Loading FLUX models...")
        
        # Determine max_length based on model name
        max_length = 256 if self.model_name == "flux-schnell" else 512
        
        # Load model components
        self.t5 = load_t5(self.device, max_length=max_length)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.model_name, device=self.device)
        self.ae = load_ae(self.model_name, device=self.device)
        
        print("✅ FLUX models loaded successfully")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        return np.array(Image.open(image_path).convert('RGB'))
    
    @torch.inference_mode()
    def encode(self, init_image: np.ndarray) -> torch.Tensor:
        """Encode image to latent space"""
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(self.device)
        init_image = self.ae.encode(init_image.to()).to(torch.bfloat16)
        return init_image
    
    @torch.inference_mode()
    def reconstruct_image(self, image_path: str, prompt: Optional[str] = None) -> Image.Image:
        """
        Reconstruct image using two sequential denoise_fireflow calls
        
        Args:
            image_path: Path to input image
            prompt: Optional text prompt (if None, uses a default)
            
        Returns:
            Reconstructed PIL Image
        """
        torch.set_grad_enabled(False)
        torch_device = torch.device(self.device)
        
        # Load and prepare image
        init_image = self.load_image(image_path)
        
        # Ensure dimensions are divisible by 16
        shape = init_image.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:new_h, :new_w, :]
        
        width, height = init_image.shape[0], init_image.shape[1]
        
        # Encode image to latent space
        init_image_encoded = self.encode(init_image)
        
        # Use default prompt if none provided
        if prompt is None:
            prompt = ""
        
        # Prepare inputs
        info = {
            'feature_path': 'feature',
            'feature': {},
            'inject_step': 25,
            'attn_mask_step': 0,
            'alpha': 0.0,
            'guidance': self.guidance,
            'artifact_data': [],
            'pe_step_addition': 0,
            'pe_step_removal': 0,
            'pe_step_distortion': 0,
            'pe_step_fusion': 0
        }
        
        inp, (patch_h, patch_w) = prepare(
            self.t5, self.clip, init_image_encoded, 
            prompt=prompt,
            info=info
        )
        
        info['patch_h'] = patch_h
        info['patch_w'] = patch_w
        
        # Get timesteps schedule
        timesteps = get_schedule(
            self.num_steps, 
            inp["img"].shape[1], 
            shift=(self.model_name != "flux-schnell")
        )
        
        t0 = time.perf_counter()
        
        # First denoise_fireflow: inverse=True (inversion to latent space)
        z, info = denoise_fireflow(
            self.model, 
            **inp, 
            timesteps=timesteps, 
            guidance=1.0,
            inverse=True, 
            info=info, 
            percentage_of_steps=1.0
        )
        
        # Update input with inverted latent
        inp["img"] = z
        
        # Recalculate timesteps for reconstruction
        timesteps = get_schedule(
            self.num_steps, 
            inp["img"].shape[1], 
            shift=(self.model_name != "flux-schnell")
        )
        
        # Second denoise_fireflow: inverse=False (reconstruction from latent)
        x, _ = denoise_fireflow(
            self.model, 
            **inp, 
            timesteps=timesteps, 
            guidance=self.guidance, 
            inverse=False, 
            info=info, 
            percentage_of_steps=1.0
        )
        
        # Unpack latents
        x = unpack(x.float(), width, height)
        
        # Decode to pixel space
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t1 = time.perf_counter()
        print(f"  Reconstruction done in {t1 - t0:.1f}s")
        
        # Convert to PIL Image
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img_array = (127.5 * (x + 1.0)).cpu().byte().numpy()
        img = Image.fromarray(img_array)
        
        # Explicitly delete intermediate tensors to free memory
        # Delete in order: free what we no longer need
        del init_image_encoded  # No longer needed after encoding
        del z  # No longer needed after updating inp["img"]
        del inp  # No longer needed after second denoise_fireflow
        del x  # No longer needed after converting to PIL
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return img
    
    def unload_models(self):
        """Unload models to free memory"""
        self.t5 = None
        self.clip = None
        self.model = None
        self.ae = None
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


def find_image_paths(parent_dir: Path) -> list:
    """
    Find all subdirectories containing real_image.png
    
    Args:
        parent_dir: Parent directory to search
        
    Returns:
        List of tuples (subdir_path, real_image_path)
    """
    tasks = []
    
    # Find all real_image.png files
    for real_image_path in parent_dir.rglob("real_image.png"):
        subdir = real_image_path.parent
        recon_image_path = subdir / "recon_image.png"
        
        tasks.append({
            'subdir': subdir,
            'real_image_path': real_image_path,
            'recon_image_path': recon_image_path
        })
    
    return tasks


def process_task_batch(
    tasks: List[dict],
    device: str,
    model_name: str,
    num_steps: int,
    guidance: float,
    use_caption: bool,
    gpu_id: int
) -> tuple:
    """
    Process a batch of tasks on a specific GPU
    
    Args:
        tasks: List of task dictionaries
        device: Device type ('cuda' or 'cpu')
        model_name: FLUX model name
        num_steps: Number of denoising steps
        guidance: Guidance scale
        use_caption: Whether to use caption from metadata
        gpu_id: GPU ID to use
        
    Returns:
        Tuple of (processed_count, failed_count)
    """
    # Set the specific GPU for this process
    if device == 'cuda':
        torch.cuda.set_device(gpu_id)
        device_str = f'cuda:{gpu_id}'
    else:
        device_str = 'cpu'
    
    # Initialize reconstructor for this GPU
    reconstructor = ImageReconstructor(
        device=device_str,
        model_name=model_name,
        num_steps=num_steps,
        guidance=guidance
    )
    
    processed = 0
    failed = 0
    
    # Process each task
    for task in tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            # Get image size before processing for error reporting
            try:
                with Image.open(task['real_image_path']) as img:
                    image_size = img.size  # (width, height)
            except Exception:
                image_size = None
            
            # Get prompt if using caption
            prompt = None
            if use_caption:
                metadata_path = task['subdir'] / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        prompt = metadata.get('image_caption', '')
            
            # Reconstruct image
            recon_image = reconstructor.reconstruct_image(
                str(task['real_image_path']),
                prompt=prompt
            )
            
            # Save reconstructed image
            recon_image.save(task['recon_image_path'])
            
            # Explicitly delete and clear memory after each image
            del recon_image
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            processed += 1
            
        except RuntimeError as e:
            # Check if it's a CUDA out of memory error
            error_str = str(e)
            if 'out of memory' in error_str.lower() or 'cuda' in error_str.lower():
                size_info = f"Image size: {image_size[0]}x{image_size[1]}" if image_size else "Image size: unknown"
                print(f"❌ [GPU {gpu_id}] CUDA OOM error processing {task['real_image_path']}: {size_info} - {error_str}")
            else:
                print(f"❌ [GPU {gpu_id}] Runtime error processing {task['real_image_path']}: {str(e)}")
            failed += 1
            # Clear cache even on error
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"❌ [GPU {gpu_id}] Error processing {task['real_image_path']}: {str(e)}")
            failed += 1
            # Clear cache even on error
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    # Cleanup
    reconstructor.unload_models()
    
    return processed, failed


def process_directory(
    parent_dir: str,
    device: str = 'cuda',
    model_name: str = 'flux-dev',
    num_steps: int = 25,
    guidance: float = 1.0,
    skip_existing: bool = True,
    use_caption: bool = False,
    num_gpus: int = None
):
    """
    Process all subdirectories and reconstruct images
    
    Args:
        parent_dir: Parent directory containing subdirectories with real_image.png
        device: Device type ('cuda' or 'cpu')
        model_name: FLUX model name
        num_steps: Number of denoising steps
        guidance: Guidance scale
        skip_existing: Whether to skip if recon_image.png already exists
        use_caption: Whether to use caption from metadata.json for reconstruction
        num_gpus: Number of GPUs to use (None = all available)
    """
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    # Find all tasks
    tasks = find_image_paths(parent_path)
    
    if not tasks:
        print(f"No real_image.png files found in {parent_dir}")
        return
    
    print(f"Found {len(tasks)} images to reconstruct")
    
    # Filter out existing if skip_existing is True
    if skip_existing:
        tasks = [task for task in tasks if not task['recon_image_path'].exists()]
        print(f"Processing {len(tasks)} images (skipping {len(find_image_paths(parent_path)) - len(tasks)} existing)")
    
    if not tasks:
        print("No images to process (all already reconstructed)")
        return
    
    # Determine number of GPUs to use
    if device == 'cuda':
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = available_gpus
        else:
            num_gpus = min(num_gpus, available_gpus)
        
        if num_gpus == 0:
            print("No GPUs available, falling back to CPU")
            device = 'cpu'
            num_gpus = 1
        else:
            print(f"Using {num_gpus} GPU(s) out of {available_gpus} available")
    else:
        num_gpus = 1
    
    # Split tasks across GPUs
    if num_gpus > 1:
        # Split tasks into chunks for each GPU
        chunk_size = len(tasks) // num_gpus
        task_chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            if i == num_gpus - 1:
                # Last chunk gets any remaining tasks
                end_idx = len(tasks)
            else:
                end_idx = (i + 1) * chunk_size
            task_chunks.append(tasks[start_idx:end_idx])
        
        print(f"Split {len(tasks)} tasks into {num_gpus} chunks: {[len(chunk) for chunk in task_chunks]}")
        
        # Create processes for each GPU
        with mp.Pool(processes=num_gpus) as pool:
            # Prepare arguments for each worker: (chunk, device, model_name, num_steps, guidance, use_caption, gpu_id)
            worker_args = [
                (chunk, device, model_name, num_steps, guidance, use_caption, gpu_id)
                for gpu_id, chunk in enumerate(task_chunks)
            ]
            
            # Map tasks to GPUs
            results = pool.starmap(process_task_batch, worker_args)
        
        # Aggregate results
        total_processed = sum(r[0] for r in results)
        total_failed = sum(r[1] for r in results)
        
    else:
        # Single GPU/CPU processing
        print("Using single GPU/CPU processing")
        reconstructor = ImageReconstructor(
            device=device,
            model_name=model_name,
            num_steps=num_steps,
            guidance=guidance
        )
        
        total_processed = 0
        total_failed = 0
        
        for task in tqdm(tasks, desc="Reconstructing images"):
            try:
                # Get image size before processing for error reporting
                try:
                    with Image.open(task['real_image_path']) as img:
                        image_size = img.size  # (width, height)
                except Exception:
                    image_size = None
                
                # Get prompt if using caption
                prompt = None
                if use_caption:
                    metadata_path = task['subdir'] / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            prompt = metadata.get('image_caption', '')
                
                # Reconstruct image
                recon_image = reconstructor.reconstruct_image(
                    str(task['real_image_path']),
                    prompt=prompt
                )
                
                # Save reconstructed image
                recon_image.save(task['recon_image_path'])
                
                # Explicitly delete and clear memory after each image
                del recon_image
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                total_processed += 1
                
            except RuntimeError as e:
                # Check if it's a CUDA out of memory error
                error_str = str(e)
                if 'out of memory' in error_str.lower() or 'cuda' in error_str.lower():
                    size_info = f"Image size: {image_size[0]}x{image_size[1]}" if image_size else "Image size: unknown"
                    print(f"❌ CUDA OOM error processing {task['real_image_path']}: {size_info} - {error_str}")
                else:
                    print(f"❌ Runtime error processing {task['real_image_path']}: {str(e)}")
                total_failed += 1
                # Clear cache even on error
                if device == 'cuda':
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"❌ Error processing {task['real_image_path']}: {str(e)}")
                total_failed += 1
                # Clear cache even on error
                if device == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        reconstructor.unload_models()
    
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)
    print(f"Total images: {len(find_image_paths(parent_path))}")
    print(f"✅ Processed: {total_processed}")
    print(f"❌ Failed: {total_failed}")
    print("="*60)


def main():
    """Main function for image reconstruction script."""
    parser = argparse.ArgumentParser(
        description='Reconstruct images using FLUX model with two-step denoising'
    )
    parser.add_argument(
        'parent_dir',
        type=str,
        help='Parent directory containing subdirectories with real_image.png files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for model inference (default: cuda)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='flux-dev',
        choices=['flux-dev', 'flux-schnell'],
        help='FLUX model to use (default: flux-dev)'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=25,
        help='Number of denoising steps (default: 25)'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=5.0,
        help='Guidance scale (default: 1.0 for reconstruction)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=False,
        help='Skip files that already have recon_image.png (default: True)'
    )
    parser.add_argument(
        '--use-caption',
        action='store_true',
        help='Use caption from metadata.json as prompt for reconstruction'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for parallel processing (default: all available)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting image reconstruction for directory: {args.parent_dir}")
    print(f"📱 Device: {args.device}")
    print(f"🤖 Model: {args.model_name}")
    print(f"🔢 Steps: {args.num_steps}")
    print(f"🎯 Guidance: {args.guidance}")
    print(f"⏭️  Skip existing: {args.skip_existing}")
    print(f"💬 Use caption: {args.use_caption}")
    if args.device == 'cuda':
        print(f"🎮 GPUs: {args.num_gpus if args.num_gpus else 'all available'}")
    print()
    
    # Process directory with multi-GPU support
    process_directory(
        parent_dir=args.parent_dir,
        device=args.device,
        model_name=args.model_name,
        num_steps=args.num_steps,
        guidance=args.guidance,
        skip_existing=args.skip_existing,
        use_caption=args.use_caption,
        num_gpus=args.num_gpus
    )


if __name__ == "__main__":
    main()
