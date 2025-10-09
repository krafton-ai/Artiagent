#!/usr/bin/env python3
"""
Script to process .pkl files containing image captions and refine them according to specific rules:
1. Keep entries where the caption contains 'To elaborate,' (unchanged)
2. Replace remaining captions with 'There are no artifacts in this image'
"""

import pickle
import os
import shutil
from pathlib import Path

def process_pkl_file(input_path, output_path):
    """
    Process a single .pkl file according to the specified rules.
    
    Args:
        input_path (str): Path to the input .pkl file
        output_path (str): Path to save the processed .pkl file
    """
    print(f"Processing {input_path}...")
    
    # Load the original data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Process the data
    processed_data = {}
    kept_count = 0
    replaced_count = 0
    
    for image_name, image_data in data.items():
        # Check if the explanation contains 'To elaborate,'
        explanation = image_data['response']['explanation']
        
        if 'To elaborate,' in explanation:
            # Keep this entry as is (don't remove it)
            processed_data[image_name] = image_data
            kept_count += 1
        else:
            # Replace the explanation with the standard message
            image_data['response']['explanation'] = 'There are no artifacts in this image'
            processed_data[image_name] = image_data
            replaced_count += 1
    
    # Save the processed data
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"  Original entries: {len(data)}")
    print(f"  Kept entries: {kept_count} (contained 'To elaborate,')")
    print(f"  Replaced entries: {replaced_count} (now say 'There are no artifacts in this image')")
    print(f"  Final entries: {len(processed_data)}")
    print(f"  Saved to: {output_path}")
    print()

def main():
    # Define input and output directories
    input_dir = Path("/data2/jhpark/image-artifacts/eval/legion_responses")
    output_dir = Path("/data2/jhpark/image-artifacts/eval/refined_legion_responses")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    print()
    
    # Find all .pkl files in the input directory
    pkl_files = list(input_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found in the input directory!")
        return
    
    print(f"Found {len(pkl_files)} .pkl files to process:")
    for pkl_file in pkl_files:
        print(f"  - {pkl_file.name}")
    print()
    
    # Process each .pkl file
    for pkl_file in pkl_files:
        output_file = output_dir / pkl_file.name
        process_pkl_file(pkl_file, output_file)
    
    print("Processing complete!")
    print(f"Refined .pkl files saved to: {output_dir}")

if __name__ == "__main__":
    main()
