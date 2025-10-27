#!/usr/bin/env python3
"""
Generate VQA dataset from multiple ArtiAgent directories.

Usage:
    python generate_vqa_multi_dir.py --input <dir1> <dir2> ... --output <output_json> [options]
    python generate_vqa_multi_dir.py --input-dirs <dir1> <dir2> ... --output <output_json> [options]
"""

import json
import argparse
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import glob

# Handle both direct execution and module import
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vqa_gen.types import ArtiInstance, ArtifactRegion
    from vqa_gen.vqa_sampler import VQASampler
    from vqa_gen.vqa_serialize import VQASerializer
    from vqa_gen.vqa_prompts import VQAPrompts
else:
    from .types import ArtiInstance, ArtifactRegion
    from .vqa_sampler import VQASampler
    from .vqa_serialize import VQASerializer
    from .vqa_prompts import VQAPrompts


def load_source_images(source_dir: str, max_images: int = None) -> List[ArtiInstance]:
    """Load real images from source directory to create balanced binary classification.
    
    Args:
        source_dir: Path to source directory containing images (supports COCO format with train2017/ subdirectory or direct image directory)
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        List of ArtiInstance objects with real images only (no artifacts)
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Try COCO format first (train2017 subdirectory), then direct image directory
    images_dir = source_path / "train2017"
    if not images_dir.exists():
        images_dir = source_path
    
    if not images_dir.exists():
        raise ValueError(f"Image directory not found: {images_dir}")
    
    # Get all image files (support common formats)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
        image_files.extend(list(images_dir.glob(ext)))
    
    if max_images:
        image_files = image_files[:max_images]
    
    instances = []
    for image_file in tqdm(image_files, desc="Loading source images"):
        # Create ArtiInstance with only real image (no artifacts)
        instance = ArtiInstance(
            real_image=str(image_file.resolve()),  # Convert to absolute path
            artifact_image=None,
            metadata_caption=None,  # Source captions not used for this purpose
            artifacts=[]  # No artifacts for real images
        )
        instances.append(instance)
    
    return instances


def load_artiagent_directories(directories: List[str], source_dir: str = None, balance_real_images: bool = True, max_instances_per_path: int = None, real_image_filename: str = "real_image.png", artifact_image_filename: str = "artifact_image.png") -> List[ArtiInstance]:
    """Load ArtiAgent data from multiple directory structures and optionally balance with real images.
    
    Args:
        directories: List of paths to ArtiAgent directories containing UUID subdirectories
        source_dir: Path to source directory for real images (optional)
        balance_real_images: Whether to balance artifact and real images
        max_instances_per_path: Maximum instances to load from each directory (None for no limit)
        real_image_filename: Filename to use for real images (default: "real_image.png")
        artifact_image_filename: Filename to use for artifact images (default: "artifact_image.png")
    
    Returns:
        List of ArtiInstance objects from all directories, optionally balanced with real images
    """
    all_instances = []
    
    # Load artifact instances
    for directory in directories:
        print(f"Loading from {directory}...")
        instances = load_artiagent_directory(directory, max_instances=max_instances_per_path, real_image_filename=real_image_filename, artifact_image_filename=artifact_image_filename)
        all_instances.extend(instances)
        print(f"  Loaded {len(instances)} instances")
    
    # Load real images for balancing if requested
    if source_dir and balance_real_images:
        print(f"Loading real images from {source_dir} for balancing...")
        real_instances = load_source_images(source_dir, max_images=len(all_instances))
        all_instances.extend(real_instances)
        print(f"  Loaded {len(real_instances)} real image instances")
        print(f"  Total instances: {len(all_instances)} (balanced)")
    
    return all_instances


def load_artiagent_directory(directory: str, max_instances: int = None, real_image_filename: str = "real_image.png", artifact_image_filename: str = "artifact_image.png") -> List[ArtiInstance]:
    """Load ArtiAgent data from directory structure.
    
    Args:
        directory: Path to ArtiAgent directory containing UUID subdirectories
        max_instances: Maximum number of instances to load (None for no limit)
        real_image_filename: Filename to use for real images (default: "real_image.png")
        artifact_image_filename: Filename to use for artifact images (default: "artifact_image.png")
    
    Returns:
        List of ArtiInstance objects
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    instances = []
    
    # Iterate through all subdirectories
    subdirs = [d for d in directory_path.iterdir() if d.is_dir()]
    
    # Limit number of subdirectories if max_instances is specified
    if max_instances is not None:
        subdirs = subdirs[:max_instances]
    
    for subdir in tqdm(subdirs, desc="Loading ArtiAgent data"):
        # Expected files in each subdirectory
        metadata_file = subdir / "metadata.json"
        real_image_file = subdir / real_image_filename
        artifact_image_file = subdir / artifact_image_filename
        
        # Skip if metadata.json doesn't exist
        if not metadata_file.exists():
            continue
        
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Parse artifacts
        artifacts = []
        if "artifacts" in metadata and metadata["artifacts"]:
            for artifact in metadata["artifacts"]:
                if "target_bbox" in artifact and "label" in artifact:
                    bbox = tuple(artifact["target_bbox"])
                    label = artifact["label"]
                    artifacts.append(ArtifactRegion(bbox=bbox, label=label))
        
        # Get caption
        caption = metadata.get("caption")
        
        # Get image paths (convert to absolute paths)
        real_image = str(real_image_file.resolve()) if real_image_file.exists() else None
        artifact_image = str(artifact_image_file.resolve()) if artifact_image_file.exists() else None
        
        # Create instance
        instance = ArtiInstance(
            real_image=real_image,
            artifact_image=artifact_image,
            metadata_caption=caption,
            artifacts=artifacts
        )
        
        instances.append(instance)
    
    return instances


def split_instances(
    instances: List[ArtiInstance],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[ArtiInstance], List[ArtiInstance]]:
    """Split instances into train and validation sets.
    
    Args:
        instances: List of ArtiInstance objects
        train_ratio: Ratio of instances for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_instances, val_instances)
    """
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError(f"Train ratio must be between 0 and 1, got {train_ratio}")
    
    random.seed(seed)
    shuffled_instances = instances.copy()
    random.shuffle(shuffled_instances)
    
    n_total = len(shuffled_instances)
    n_train = int(n_total * train_ratio)
    
    train_instances = shuffled_instances[:n_train]
    val_instances = shuffled_instances[n_train:]
    
    return train_instances, val_instances


def generate_vqa_dataset(
    instances: List[ArtiInstance],
    format_dropout: float = 0.15,
    seed: int = 42,
    sample_one_mode: bool = False
) -> List[Dict[str, Any]]:
    """Generate VQA dataset from ArtiAgent instances.
    
    Args:
        instances: List of ArtiInstance objects
        format_dropout: Probability to omit format instructions
        seed: Random seed for reproducibility
    
    Returns:
        List of conversation dictionaries
    """
    random.seed(seed)
    
    sampler = VQASampler(
        format_dropout=format_dropout,
        qa_dropout_rate=0.0,  # No longer used (kept for compatibility)
        single_turn_prob=0.0  # No longer used (kept for compatibility)
    )
    
    conversations = []
    
    for instance in tqdm(instances, desc="Generating VQA data"):
        # Skip if no valid images
        if not instance.real_image and not instance.artifact_image:
            continue
        
        # Determine available modes based on available images
        has_real = instance.real_image is not None
        has_artifact = instance.artifact_image is not None and len(instance.artifacts) > 0
        
        # Generate all possible modes for this instance
        modes_to_generate = []
        
        if has_real:
            modes_to_generate.append("real")
        if has_artifact:
            modes_to_generate.append("artifact")
        if has_real and has_artifact:
            modes_to_generate.append("pair")
        
        # Skip if no valid modes
        if not modes_to_generate:
            continue
        
        # Decide which modes to generate
        if sample_one_mode and modes_to_generate:
            selected_modes = [random.choice(modes_to_generate)]
        else:
            selected_modes = modes_to_generate

        # Generate conversations for selected modes
        for mode in selected_modes:
            images, qa_pairs = sampler.sample_conversation(instance, mode=mode)
            if not qa_pairs:
                raise ValueError(f"No Q-A pairs generated for mode: {mode}")
            conversation = VQASerializer.serialize_conversation(images, qa_pairs)
            conversations.append(conversation)
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Generate VQA dataset from multiple ArtiAgent directories")
    
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more paths to ArtiAgent directories (containing UUID subdirectories)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output VQA JSON file"
    )
    parser.add_argument(
        "--format-dropout",
        type=float,
        default=0.0,
        help="Probability to omit format instructions (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Ratio of data for training (default: 1.0, no validation split). Set < 1.0 to create train/val split."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        help="Path to source directory for real images (for balancing binary classification)"
    )
    parser.add_argument(
        "--sample-one-mode",
        action="store_true",
        help="If set, sample only one mode (real/artifact/pair) per instance; default generates all available modes"
    )
    parser.add_argument(
        "--enable-prompt-variants",
        action="store_true",
        help="Enable random selection among prompt variants (default off: always use first)"
    )
    parser.add_argument(
        "--max-instances-per-path",
        type=int,
        help="Maximum number of instances to collect from each input directory (default: no limit)"
    )
    parser.add_argument(
        "--real-image-filename",
        type=str,
        default="real_image.png",
        help="Filename to use for real images in subdirectories (default: real_image.png)"
    )
    parser.add_argument(
        "--artifact-image-filename",
        type=str,
        default="artifact_image.png",
        help="Filename to use for artifact images in subdirectories (default: artifact_image.png)"
    )
    
    args = parser.parse_args()
    
    # Get input directories
    input_dirs = args.input
    
    # Validate directories exist
    for directory in input_dirs:
        if not Path(directory).exists():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)
    
    # Load input data from all directories
    print(f"Loading ArtiAgent data from {len(input_dirs)} directories...")

    # Configure prompt variant behavior
    VQAPrompts.USE_VARIANTS = bool(args.enable_prompt_variants)

    instances = load_artiagent_directories(
        input_dirs,
        source_dir=args.source_dir,
        balance_real_images=bool(args.source_dir),
        max_instances_per_path=args.max_instances_per_path,
        real_image_filename=args.real_image_filename,
        artifact_image_filename=args.artifact_image_filename
    )
    
    print(f"Total loaded: {len(instances)} instances")
    
    # Split into train/val if train ratio < 1.0
    if args.train_ratio < 1.0:
        val_ratio = 1.0 - args.train_ratio
        print(f"Splitting data: {args.train_ratio:.1%} train, {val_ratio:.1%} validation...")
        train_instances, val_instances = split_instances(
            instances=instances,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        print(f"Train: {len(train_instances)} instances, Val: {len(val_instances)} instances")
        
        # Generate train dataset
        print("Generating train VQA dataset...")
        train_conversations = generate_vqa_dataset(
            instances=train_instances,
            format_dropout=args.format_dropout,
            seed=args.seed,
            sample_one_mode=args.sample_one_mode
        )
        
        # Generate val dataset
        print("Generating validation VQA dataset...")
        val_conversations = generate_vqa_dataset(
            instances=val_instances,
            format_dropout=args.format_dropout,
            seed=args.seed + 1,  # Different seed for val
            sample_one_mode=args.sample_one_mode
        )
        
        print(f"Generated {len(train_conversations)} train conversations")
        print(f"Generated {len(val_conversations)} validation conversations")
        
        # Save train and val files
        train_output = args.output.replace('.json', '_train.json')
        val_output = args.output.replace('.json', '_val.json')
        
        print(f"Saving train data to {train_output}...")
        VQASerializer.save_to_json(
            conversations=train_conversations,
            output_path=train_output,
            validate=True
        )
        
        print(f"Saving validation data to {val_output}...")
        VQASerializer.save_to_json(
            conversations=val_conversations,
            output_path=val_output,
            validate=True
        )
        
    else:
        # Generate single dataset (no split)
        print("Generating VQA dataset (no train/val split)...")
        conversations = generate_vqa_dataset(
            instances=instances,
            format_dropout=args.format_dropout,
            seed=args.seed,
            sample_one_mode=args.sample_one_mode
        )
        
        print(f"Generated {len(conversations)} conversations")
        
        # Save to output file
        print(f"Saving to {args.output}...")
        VQASerializer.save_to_json(
            conversations=conversations,
            output_path=args.output,
            validate=True
        )
    
    print("Done!")


if __name__ == "__main__":
    main()

