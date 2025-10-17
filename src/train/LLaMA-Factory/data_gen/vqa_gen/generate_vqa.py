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

# Handle both direct execution and module import
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vqa_gen.types import ArtiInstance, ArtifactRegion
    from vqa_gen.vqa_sampler import VQASampler
    from vqa_gen.vqa_serialize import VQASerializer
else:
    from .types import ArtiInstance, ArtifactRegion
    from .vqa_sampler import VQASampler
    from .vqa_serialize import VQASerializer


def load_artiagent_directories(directories: List[str]) -> List[ArtiInstance]:
    """Load ArtiAgent data from multiple directory structures.
    
    Args:
        directories: List of paths to ArtiAgent directories containing UUID subdirectories
    
    Returns:
        List of ArtiInstance objects from all directories
    """
    all_instances = []
    
    for directory in directories:
        print(f"Loading from {directory}...")
        instances = load_artiagent_directory(directory)
        all_instances.extend(instances)
        print(f"  Loaded {len(instances)} instances")
    
    return all_instances


def load_artiagent_directory(directory: str) -> List[ArtiInstance]:
    """Load ArtiAgent data from directory structure.
    
    Args:
        directory: Path to ArtiAgent directory containing UUID subdirectories
    
    Returns:
        List of ArtiInstance objects
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    instances = []
    
    # Iterate through all subdirectories
    subdirs = [d for d in directory_path.iterdir() if d.is_dir()]
    
    for subdir in tqdm(subdirs, desc="Loading ArtiAgent data"):
        # Expected files in each subdirectory
        metadata_file = subdir / "metadata.json"
        real_image_file = subdir / "real_image.png"
        artifact_image_file = subdir / "artifact_image.png"
        
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
        
        # Get image paths (use absolute or relative paths)
        real_image = str(real_image_file) if real_image_file.exists() else None
        artifact_image = str(artifact_image_file) if artifact_image_file.exists() else None
        
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
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[ArtiInstance], List[ArtiInstance]]:
    """Split instances into train and validation sets.
    
    Args:
        instances: List of ArtiInstance objects
        train_ratio: Ratio of instances for training (default: 0.8)
        val_ratio: Ratio of instances for validation (default: 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_instances, val_instances)
    """
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train ratio ({train_ratio}) + val ratio ({val_ratio}) must equal 1.0")
    
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
    seed: int = 42
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
        
        # Sample conversation
        images, qa_pairs = sampler.sample_conversation(instance, mode="auto")
        
        # Skip if no Q-A pairs generated
        if not qa_pairs:
            continue
        
        # Serialize to JSON format
        conversation = VQASerializer.serialize_conversation(images, qa_pairs)
        conversations.append(conversation)
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Generate VQA dataset from multiple ArtiAgent directories")
    
    # Create mutually exclusive group for input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        nargs="+",
        help="One or more paths to ArtiAgent directories (containing UUID subdirectories)"
    )
    input_group.add_argument(
        "--input-dirs",
        nargs="+",
        help="Alternative flag for multiple ArtiAgent directories"
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
        default=0.15,
        help="Probability to omit format instructions (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation before saving"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of data for validation (default: 0.2)"
    )
    parser.add_argument(
        "--split-outputs",
        action="store_true",
        help="Generate separate train and validation files"
    )
    
    args = parser.parse_args()
    
    # Determine input directories
    if args.input:
        input_dirs = args.input
    else:
        input_dirs = args.input_dirs
    
    # Validate directories exist
    for directory in input_dirs:
        if not Path(directory).exists():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)
    
    # Load input data from all directories
    print(f"Loading ArtiAgent data from {len(input_dirs)} directories...")
    instances = load_artiagent_directories(input_dirs)
    
    print(f"Total loaded: {len(instances)} instances")
    
    # Split into train/val if requested
    if args.split_outputs:
        print(f"Splitting data: {args.train_ratio:.1%} train, {args.val_ratio:.1%} validation...")
        train_instances, val_instances = split_instances(
            instances=instances,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        print(f"Train: {len(train_instances)} instances, Val: {len(val_instances)} instances")
        
        # Generate train dataset
        print("Generating train VQA dataset...")
        train_conversations = generate_vqa_dataset(
            instances=train_instances,
            format_dropout=args.format_dropout,
            seed=args.seed
        )
        
        # Generate val dataset
        print("Generating validation VQA dataset...")
        val_conversations = generate_vqa_dataset(
            instances=val_instances,
            format_dropout=args.format_dropout,
            seed=args.seed + 1  # Different seed for val
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
            validate=not args.no_validate
        )
        
        print(f"Saving validation data to {val_output}...")
        VQASerializer.save_to_json(
            conversations=val_conversations,
            output_path=val_output,
            validate=not args.no_validate
        )
        
    else:
        # Generate single dataset
        print("Generating VQA dataset...")
        conversations = generate_vqa_dataset(
            instances=instances,
            format_dropout=args.format_dropout,
            seed=args.seed
        )
        
        print(f"Generated {len(conversations)} conversations")
        
        # Save to output file
        print(f"Saving to {args.output}...")
        VQASerializer.save_to_json(
            conversations=conversations,
            output_path=args.output,
            validate=not args.no_validate
        )
    
    print("Done!")


if __name__ == "__main__":
    main()

