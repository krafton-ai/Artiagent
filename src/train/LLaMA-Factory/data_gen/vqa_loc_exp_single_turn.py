"""
Localization + Explanation Data JSON Generator (Positive Data Only)

This script generates training data for VLM models focused on artifact localization and explanation.
It processes only positive data (images with artifacts) and creates structured JSON responses
that include normalized bounding box coordinates and explanations.

The output format includes:
1. Normalized bounding boxes [0,1] range
2. Short explanations for each artifact
3. Scene captions
4. Structured JSON response format
"""

import json
import os
import math
import argparse
from PIL import Image
import random
from typing import List, Dict, Tuple

DATA_DIR = "/data2/jhpark/image-artifacts/data/"
JSON_DIR = "/home/jhpark/image-artifacts/src/train/LLaMA-Factory/data"


def create_localization_entry(img_path: str, bboxes: List[List[float]], artifacts_data: List[Dict], 
                            caption: str = "", image_width: int = None, image_height: int = None) -> Dict:
    """
    Create a dataset entry for localization and explanation (positive data only).
    
    Args:
        img_path: Path to the artifact image
        bboxes: List of bounding boxes [x_min, y_min, x_max, y_max]
        artifacts_data: List of artifact metadata dictionaries
        caption: Image caption
        image_width: Image width (optional)
        image_height: Image height (optional)
    
    Returns:
        Dictionary containing the dataset entry
    """
    try:
        # Normalize bboxes to [0,1] range if image dimensions are provided
        normalized_bboxes = []
        explanations = []
        
        if image_width and image_height:
            for i, artifact in enumerate(artifacts_data):
                if i < len(bboxes) and len(bboxes[i]) == 4:
                    bbox = bboxes[i]
                    # Normalize bbox coordinates to [0,1]
                    x1, y1, x2, y2 = bbox
                    norm_x1 = x1 / image_width
                    norm_y1 = y1 / image_height
                    norm_x2 = x2 / image_width
                    norm_y2 = y2 / image_height
                    
                    # Ensure coordinates are within [0,1] range
                    norm_x1 = int(max(0, min(1, norm_x1)))
                    norm_y1 = int(max(0, min(1, norm_y1)))
                    norm_x2 = int(max(0, min(1, norm_x2)))
                    norm_y2 = int(max(0, min(1, norm_y2)))
                    
                    normalized_bboxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
                    
                    # Get explanation from artifact data
                    explanation = artifact.get("explanation", artifact.get("label", "Visual artifact detected"))
                    explanations.append(explanation)
        
        # Create artifacts array with pixel coordinates
        artifacts_array = []
        for i, artifact in enumerate(artifacts_data):
            if i < len(bboxes) and len(bboxes[i]) == 4:
                bbox = bboxes[i]  # Use original pixel coordinates
                int_bbox = [int(round(coord)) for coord in bbox]
                explanation = artifact.get("explanation", artifact.get("label", "Visual artifact detected"))
                artifacts_array.append({
                    "bbox_2d": int_bbox,
                    "label": explanation
                })
        
        # Create the two fenced JSON blocks
        first_block = json.dumps(artifacts_array, indent=2, ensure_ascii=False)
        second_block = json.dumps({"explanation": caption}, indent=2, ensure_ascii=False)
        
        # Combine into fenced JSON blocks
        gpt_response_str = f"```json\n{first_block}\n```\n\n```json\n{second_block}\n```"
        
        # Validate JSON structure of individual blocks
        try:
            json.loads(first_block)
            json.loads(second_block)
        except json.JSONDecodeError as e:
            print(f"    ⚠️  Invalid JSON generated: {e}")
            raise
        
        entry = {
            "images": [img_path],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nAnalyze this image carefully and identify any visual artifacts present.\n\nYou must respond with exactly two fenced JSON blocks in this order:\n\nFirst JSON block - Array of artifacts with pixel coordinates:\n```json\n[\n  {\"bbox_2d\": [x1, y1, x2, y2], \"label\": description of the artifact in this region},\n  {\"bbox_2d\": [x1, y1, x2, y2], \"label\": description of the artifact in this region}\n]\n```\n\nSecond JSON block - Explanation:\n```json\n{\"explanation\": description of the anomalies in this image.}\n```\n\nRequirements:\n- Use pixel coordinates (not normalized)\n- Each bbox_2d array must have exactly 4 numbers: [x_min, y_min, x_max, y_max]\n- Provide explanations in English only\n- Ensure both JSON blocks are properly formatted and valid.\n"
                },
                {
                    "from": "gpt",
                    "value": gpt_response_str
                }
            ],
            "has_artifacts": True,
            "artifact_count": len(artifacts_array)
        }
        
        # Add image dimensions if available
        if image_width and image_height:
            entry["image_dimensions"] = {"width": image_width, "height": image_height}
        
        return entry
        
    except Exception as e:
        print(f"    ⚠️  Error creating localization entry: {e}")
        raise

def safe_process_artifact_image(image_path: str, metadata_path: str, dataset: List[Dict], 
                              stats: Dict, image_width: int = None, image_height: int = None) -> bool:
    """
    Safely process an artifact image for localization training.
    
    Args:
        image_path: Path to the artifact image
        metadata_path: Path to the metadata JSON file
        dataset: List to append the processed entry to
        stats: Statistics dictionary to update
        image_width: Image width (optional)
        image_height: Image height (optional)
    
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Validate input files
        if not os.path.exists(image_path):
            print(f"  ⚠️  Input image does not exist: {image_path}")
            return False
        
        if not os.path.exists(metadata_path):
            print(f"  ⚠️  Metadata does not exist: {metadata_path}")
            return False
        
        # Check file size (avoid processing corrupted/empty files)
        file_size = os.path.getsize(image_path)
        if file_size < 1000:  # Less than 1KB is likely corrupted
            print(f"  ⚠️  Input image too small ({file_size} bytes): {image_path}")
            return False
        
        if file_size > 50 * 1024 * 1024:  # Greater than 50MB
            print(f"  ⚠️  Input image too large ({file_size // 1024 // 1024} MB): {image_path}")
            return False
        
        # Load and validate image
        try:
            with Image.open(image_path) as img:
                img.load()  # Force loading to catch corruption early
                
                if img.size[0] < 16 or img.size[1] < 16:
                    print(f"  ⚠️  Image too small {img.size}: {image_path}")
                    return False
                
                if img.size[0] > 4096 or img.size[1] > 4096:
                    print(f"  ⚠️  Image too large {img.size}: {image_path}")
                    return False
                
                img = img.convert('RGB')
                actual_width, actual_height = img.size
                
        except Exception as e:
            print(f"  ⚠️  Cannot load image {image_path}: {e}")
            return False
        
        # Load metadata
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            artifacts = metadata.get("artifacts", [])
            caption = metadata.get("caption", "An image with artifacts")
            
            # Extract and validate bboxes
            bboxes = []
            valid_artifacts = []
            
            for artifact in artifacts:
                bbox = artifact.get("target_bbox")
                if bbox and len(bbox) == 4:
                    try:
                        bbox = [float(coord) for coord in bbox]
                        if all(math.isfinite(coord) for coord in bbox):
                            # Validate bbox coordinates
                            x_min, y_min, x_max, y_max = bbox
                            if (0 <= x_min < x_max <= actual_width and 
                                0 <= y_min < y_max <= actual_height):
                                bboxes.append(bbox)
                                valid_artifacts.append(artifact)
                    except (ValueError, TypeError):
                        continue
            
            # Skip if no valid artifacts
            if not valid_artifacts:
                print(f"  ⚠️  No valid artifacts found in {image_path}")
                return False
                
        except Exception as e:
            print(f"  ⚠️  Error loading metadata: {e}")
            return False
        
        # Create dataset entry
        try:
            entry = create_localization_entry(
                image_path, 
                bboxes, 
                valid_artifacts,
                caption,
                actual_width,
                actual_height
            )
            dataset.append(entry)
            return True
            
        except Exception as e:
            print(f"  ⚠️  Failed to create dataset entry: {e}")
            return False
        
    except Exception as e:
        print(f"  ⚠️  Error processing artifact image: {e}")
        return False

def process_artifact_sources(artifact_sources: List[Dict], dataset: List[Dict], 
                           stats: Dict, max_samples_per_source: int = None) -> int:
    """
    Process artifact image sources for localization training.
    
    Args:
        artifact_sources: List of source dictionaries with 'path' and 'name' keys
        dataset: List to append processed entries to
        stats: Statistics dictionary to update
        max_samples_per_source: Maximum samples to process per source
    
    Returns:
        Number of samples successfully added
    """
    print(f"🔄 Processing Artifact Sources for Localization Training")
    print("=" * 60)
    
    total_processed = 0
    total_added = 0
    
    for source in artifact_sources:
        if not os.path.exists(source['path']):
            print(f"  ⚠️  Source path does not exist: {source['path']}")
            continue
            
        print(f"\n🔄 Processing {source['name']}")
        initial_dataset_size = len(dataset)
        processed_count = 0
        added_count = 0
        
        try:
            for root, dirs, files in os.walk(source['path']):
                # Shuffle directories to get random samples
                random.shuffle(dirs)
                
                for img_id in dirs:
                    if max_samples_per_source and added_count >= max_samples_per_source:
                        break
                        
                    dir_path = os.path.join(root, img_id)
                    metadata_path = os.path.join(dir_path, "metadata.json")
                    artifact_image_path = os.path.join(dir_path, "artifact_image.png")
                    
                    # Check if both files exist
                    if not (os.path.exists(artifact_image_path) and os.path.exists(metadata_path)):
                        continue
                    
                    # Process the artifact image
                    if safe_process_artifact_image(artifact_image_path, metadata_path, dataset, stats):
                        added_count += 1
                    
                    processed_count += 1
                    
                    if processed_count % 50 == 0 and processed_count > 0:
                        print(f"    📊 Progress: {processed_count} processed, {added_count} added")
                
                # Break outer loop if we've reached the limit
                if max_samples_per_source and added_count >= max_samples_per_source:
                    break
        
        except Exception as e:
            print(f"  ❌ Error processing {source['name']}: {e}")
        
        final_added = len(dataset) - initial_dataset_size
        total_added += final_added
        total_processed += processed_count
        
        print(f"  ✅ Added {final_added} localization samples from {source['name']}")
    
    print(f"✅ Artifact sources processed: {total_added} samples")
    return total_added

def main(target_dataset_size: int = 1000, output_name: str = None):
    """
    Generate localization + explanation training dataset (positive data only).
    
    Args:
        target_dataset_size: Target number of positive samples to generate
        output_name: Custom output filename (without extension)
    """
    
    print("🎯 Localization + Explanation Dataset Generator (Positive Data Only)")
    print("=" * 60)
    print(f"🎯 Target dataset size: {target_dataset_size} (positive samples only)")
    print("=" * 60)
    
    dataset = []
    stats = {'processed_count': 0, 'successful_count': 0, 'failed_count': 0}

    # Data sources - only positive artifact data
    artifact_sources = [
        {
            'path': os.path.join(DATA_DIR, "train/vanilla/filtered_animals_1k"),
            'name': "coco_animal_vanilla"
        },
        {
            'path': os.path.join(DATA_DIR, "train/vanilla/filtered_person_1k"),
            'name': "coco_person_vanilla"
        }
    ]
    
    # Filter available sources
    available_sources = [s for s in artifact_sources if os.path.exists(s['path'])]
    
    if not available_sources:
        print("⚠️  No artifact sources available!")
        return
    
    print(f"📊 Available sources: {len(available_sources)}")
    for source in available_sources:
        print(f"   - {source['name']}: {source['path']}")
    
    # Distribute samples across available sources
    samples_per_source = target_dataset_size // len(available_sources)
    remainder = target_dataset_size % len(available_sources)
    
    for i, source in enumerate(available_sources):
        source['max_samples'] = samples_per_source + (1 if i < remainder else 0)
        print(f"   {source['name']}: {source['max_samples']} samples")
    
    # Process artifact sources
    total_added = process_artifact_sources(available_sources, dataset, stats)
    
    # Ensure we don't exceed the target size
    if len(dataset) > target_dataset_size:
        print(f"\n⚠️  Dataset size ({len(dataset)}) exceeds target ({target_dataset_size}). Truncating...")
        dataset = dataset[:target_dataset_size]
        print(f"✅ Dataset truncated to {len(dataset)} samples")
    
    # Validate dataset
    print(f"\n🔍 Dataset Validation:")
    print("=" * 40)
    actual_count = len(dataset)
    artifact_counts = [entry.get('artifact_count', 0) for entry in dataset]
    total_artifacts = sum(artifact_counts)
    avg_artifacts = total_artifacts / actual_count if actual_count > 0 else 0
    
    print(f"  Target samples: {target_dataset_size}")
    print(f"  Actual samples: {actual_count}")
    print(f"  Total artifacts: {total_artifacts}")
    print(f"  Average artifacts per image: {avg_artifacts:.2f}")
    
    if actual_count == 0:
        print("⚠️  Warning: No samples were generated!")
        return
    
    # Shuffle the final dataset
    print(f"\n🔀 Shuffling final dataset...")
    random.shuffle(dataset)
    
    # Save with validation
    if output_name:
        output_file = os.path.join(JSON_DIR, f"{output_name}.json")
    else:
        output_file = os.path.join(JSON_DIR, f"artifact_localization_positive_{target_dataset_size}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Dataset saved: {output_file}")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"  Target dataset size: {target_dataset_size}")
    print(f"  Actual dataset size: {actual_count}")
    print(f"  Success rate: {(actual_count/target_dataset_size)*100:.1f}%")
    print(f"  Total artifacts detected: {total_artifacts}")
    print(f"  Average artifacts per image: {avg_artifacts:.2f}")
    
    # Show sample entries
    print(f"\n📋 Sample Dataset Entries:")
    print("=" * 60)
    for i, entry in enumerate(dataset[:2]):
        print(f"Sample {i+1}:")
        print(f"  Image: {entry['images'][0]}")
        print(f"  Artifact count: {entry.get('artifact_count', 'N/A')}")
        print(f"  Response preview: {entry['conversations'][1]['value'][:200]}...")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate localization + explanation training dataset (positive data only)")
    parser.add_argument(
        "--target_size", 
        type=int, 
        default=1000,
        help="Target number of positive samples to generate"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom output filename (without extension). If not provided, uses 'artifact_localization_positive_{target_size}'"
    )
    
    args = parser.parse_args()
    
    main(target_dataset_size=args.target_size, output_name=args.output_name)