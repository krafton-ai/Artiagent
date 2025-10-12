#!/usr/bin/env python3
"""
Script to split a JSON dataset into train and validation sets.

Usage:
    python train_val_split.py <input_json_file> <train_ratio> [options]

Example:
    python train_val_split.py artifact_1k_vanilla.json 0.8
    python train_val_split.py artifact_1k_vanilla.json 0.8 --shuffle --seed 42
"""

import json
import argparse
import random
import os
from pathlib import Path


def split_json_dataset(input_file, train_ratio, shuffle=True, seed=None, output_dir=None, balance_val=True):
    """
    Split a JSON dataset into train and validation sets.
    
    Args:
        input_file (str): Path to input JSON file
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
        shuffle (bool): Whether to shuffle the data before splitting
        seed (int): Random seed for reproducibility
        output_dir (str): Directory to save output files (default: same as input)
        balance_val (bool): Whether to balance positive and negative samples in validation set
    
    Returns:
        tuple: (train_file_path, val_file_path)
    """
    # Validate inputs
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0.0 and 1.0")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Load the JSON data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects")
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    if balance_val:
        # Separate positive and negative samples
        positive_samples = [item for item in data if not item.get('negative_sample', False)]
        negative_samples = [item for item in data if item.get('negative_sample', False)]
        
        print(f"Positive samples: {len(positive_samples)}")
        print(f"Negative samples: {len(negative_samples)}")
        
        # Shuffle each group if requested
        if shuffle:
            print("Shuffling data...")
            random.shuffle(positive_samples)
            random.shuffle(negative_samples)
        
        # Calculate split sizes for each group
        pos_train_size = int(len(positive_samples) * train_ratio)
        neg_train_size = int(len(negative_samples) * train_ratio)
        
        # For balanced validation, use the smaller group size
        min_val_size = min(len(positive_samples) - pos_train_size, len(negative_samples) - neg_train_size)
        
        # Adjust train sizes to ensure balanced validation
        pos_train_size = len(positive_samples) - min_val_size
        neg_train_size = len(negative_samples) - min_val_size
        
        print(f"Balanced validation set size: {min_val_size * 2} (equal positive and negative)")
        print(f"Train samples: {pos_train_size + neg_train_size}")
        print(f"  - Positive train: {pos_train_size}")
        print(f"  - Negative train: {neg_train_size}")
        print(f"  - Positive val: {min_val_size}")
        print(f"  - Negative val: {min_val_size}")
        
        # Split the data
        train_data = positive_samples[:pos_train_size] + negative_samples[:neg_train_size]
        val_data = positive_samples[pos_train_size:pos_train_size + min_val_size] + negative_samples[neg_train_size:neg_train_size + min_val_size]
        
        # Shuffle the final train and val sets if requested
        if shuffle:
            random.shuffle(train_data)
            random.shuffle(val_data)
    
    else:
        # Original splitting logic without balancing
        if shuffle:
            print("Shuffling data...")
            random.shuffle(data)
        
        # Calculate split indices
        train_size = int(total_samples * train_ratio)
        val_size = total_samples - train_size
        
        print(f"Train samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Split the data
        train_data = data[:train_size]
        val_data = data[train_size:]
    
    # Determine output directory and filenames
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    input_path = Path(input_file)
    base_name = input_path.stem
    suffix = input_path.suffix
    
    train_file = os.path.join(output_dir, f"{base_name}_train{suffix}")
    val_file = os.path.join(output_dir, f"{base_name}_val{suffix}")
    
    # Save train data
    print(f"Saving train data to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save validation data
    print(f"Saving validation data to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print("Split completed successfully!")
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSON dataset into train and validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_val_split.py artifact_1k_vanilla.json 0.8
  python train_val_split.py artifact_1k_vanilla.json 0.8 --shuffle --seed 42
  python train_val_split.py artifact_1k_vanilla.json 0.7 --no-shuffle --output-dir ./splits
  python train_val_split.py artifact_1k_vanilla.json 0.8 --no-balance  # Disable validation balancing
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input JSON file'
    )
    
    parser.add_argument(
        'train_ratio',
        type=float,
        help='Ratio of data to use for training (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='Shuffle data before splitting (default: True)'
    )
    
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle data before splitting'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Directory to save output files (default: same as input file)'
    )
    
    parser.add_argument(
        '--no-balance',
        action='store_true',
        help='Do not balance positive and negative samples in validation set'
    )
    
    args = parser.parse_args()
    
    # Handle shuffle options
    shuffle = args.shuffle and not args.no_shuffle
    balance_val = not args.no_balance
    
    try:
        train_file, val_file = split_json_dataset(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            shuffle=shuffle,
            seed=args.seed,
            output_dir=args.output_dir,
            balance_val=balance_val
        )
        
        print(f"\nOutput files:")
        print(f"  Train: {train_file}")
        print(f"  Validation: {val_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
