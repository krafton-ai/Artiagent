
"""
Image Captioning Script

This script processes directories containing metadata.json files and adds
image_caption field using the BLIP2 model.

Usage:
    python image_captioning.py /path/to/parent/directory
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def load_blip2_model(device: str = "cuda"):
    """Load BLIP2 model and processor."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        device_map="auto" if device == "cuda" else None
    )
    return processor, model


def generate_captions_batch(
    image_paths: list,
    processor: Blip2Processor, 
    model: Blip2ForConditionalGeneration, 
    device: str = "cuda"
) -> list:
    """
    Generate captions for multiple images in batch using BLIP2 model.
    
    Args:
        image_paths: List of paths to image files
        processor: BLIP2 processor instance
        model: BLIP2 model instance
        device: Device to use (cuda/cpu)
        
    Returns:
        List of generated caption strings
    """
    # Load all images
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    
    # Process images in batch
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    # Generate captions in batch
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=50)
    
    # Decode captions
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    
    return captions


def find_image_path(subdir: Path, metadata: dict) -> Optional[str]:
    """
    Find the path to the real image from metadata or directory.
    
    Args:
        subdir: Path to the subdirectory containing metadata.json
        metadata: Loaded metadata dictionary
        
    Returns:
        Path to the image file, or None if not found
    """
    # First check if real_image_path is in metadata
    if 'real_image_path' in metadata:
        image_path = metadata['real_image_path']
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        # If relative, try relative to subdir
        relative_path = os.path.join(subdir, image_path)
        if os.path.exists(relative_path):
            return relative_path
    
    # Check for common image filenames
    common_filenames = ['real_image.png', 'real_image.jpg', 'image.png', 'image.jpg']
    for filename in common_filenames:
        image_path = subdir / filename
        if image_path.exists():
            return str(image_path)
    
    return None


def process_directory(
    parent_dir: str,
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    device: str = "cuda",
    overwrite: bool = False,
    skip_existing: bool = True,
    batch_size: int = 8
):
    """
    Process all metadata.json files in subdirectories and add image_caption.
    Processes images in batches for efficiency.
    
    Args:
        parent_dir: Parent directory containing subdirectories with metadata.json
        processor: BLIP2 processor instance
        model: BLIP2 model instance
        device: Device to use (cuda/cpu)
        overwrite: Whether to overwrite existing image_caption fields
        skip_existing: Whether to skip files that already have image_caption
        batch_size: Number of images to process in each batch
    """
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    # Find all metadata.json files
    metadata_files = list(parent_path.rglob("metadata.json"))
    
    if not metadata_files:
        print(f"No metadata.json files found in {parent_dir}")
        return
    
    print(f"Found {len(metadata_files)} metadata.json files")
    print(f"Using batch size: {batch_size}")
    
    # Collect all tasks first
    tasks = []
    for metadata_file in metadata_files:
        try:
            subdir = metadata_file.parent
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if image_caption already exists
            if 'image_caption' in metadata and skip_existing and not overwrite:
                continue
            
            # Find image path
            image_path = find_image_path(subdir, metadata)
            if not image_path:
                continue
            
            tasks.append({
                'metadata_file': metadata_file,
                'metadata': metadata,
                'image_path': image_path
            })
        except Exception as e:
            print(f"⚠️  Error preparing {metadata_file}: {str(e)}")
            continue
    
    if not tasks:
        print("No files to process (all already have captions or errors occurred)")
        return
    
    print(f"Processing {len(tasks)} files in batches of {batch_size}")
    
    processed = 0
    skipped = len(metadata_files) - len(tasks)
    failed = 0
    
    # Process in batches
    for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
        batch = tasks[i:i + batch_size]
        batch_image_paths = [task['image_path'] for task in batch]
        
        try:
            # Generate captions for the batch
            captions = generate_captions_batch(batch_image_paths, processor, model, device)
            
            # Update metadata files with captions
            for task, caption in zip(batch, captions):
                try:
                    task['metadata']['image_caption'] = caption
                    
                    # Save updated metadata
                    with open(task['metadata_file'], 'w', encoding='utf-8') as f:
                        json.dump(task['metadata'], f, indent=2, ensure_ascii=False)
                    
                    processed += 1
                except Exception as e:
                    print(f"❌ Error saving {task['metadata_file']}: {str(e)}")
                    failed += 1
                    
        except Exception as e:
            print(f"❌ Error processing batch starting at {i}: {str(e)}")
            # Fallback to individual processing for this batch
            for task in batch:
                try:
                    # Generate caption individually
                    img = Image.open(task['image_path']).convert("RGB")
                    inputs = processor(img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_length=50)
                    caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                    
                    task['metadata']['image_caption'] = caption
                    with open(task['metadata_file'], 'w', encoding='utf-8') as f:
                        json.dump(task['metadata'], f, indent=2, ensure_ascii=False)
                    
                    processed += 1
                except Exception as e2:
                    print(f"❌ Error processing {task['metadata_file']}: {str(e2)}")
                    failed += 1
    
    print("\n" + "="*60)
    print("CAPTIONING SUMMARY")
    print("="*60)
    print(f"Total files: {len(metadata_files)}")
    print(f"✅ Processed: {processed}")
    print(f"⏭️  Skipped (already had caption): {skipped}")
    print(f"❌ Failed: {failed}")
    print("="*60)


def main():
    """Main function for image captioning script."""
    parser = argparse.ArgumentParser(
        description='Generate image captions using BLIP2 model for metadata.json files'
    )
    parser.add_argument(
        'parent_dir',
        type=str,
        help='Parent directory containing subdirectories with metadata.json files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for model inference (default: cuda)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing image_caption fields if they already exist'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=False,
        help='Skip files that already have image_caption field (default: True)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing images (default: 8)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting image captioning for directory: {args.parent_dir}")
    print(f"📱 Device: {args.device}")
    print(f"🔄 Overwrite existing: {args.overwrite}")
    print(f"⏭️  Skip existing: {args.skip_existing}")
    print(f"📦 Batch size: {args.batch_size}")
    print()
    
    # Load BLIP2 model
    print("Loading BLIP2 model...")
    processor, model = load_blip2_model(args.device)
    print("✅ BLIP2 model loaded successfully")
    print()
    
    # Process directory
    process_directory(
        args.parent_dir,
        processor,
        model,
        device=args.device,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()