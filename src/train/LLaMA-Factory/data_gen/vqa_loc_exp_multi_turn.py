"""
Multi-Turn VQA Localization + Explanation Data JSON Generator (Positive Data Only)

This script generates multi-turn training data for VLM models focused on artifact localization and explanation.
It processes only positive data (images with artifacts) and creates structured multi-turn conversations
that progressively extract artifact information.

The conversation flow:
1. Q1: Detection + Bounding boxes (bbox coordinates only)
2. Q2: Individual explanations for each bbox
3. Q3: Overall scene description

All images are assumed to have artifacts (no negative samples).
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

# Multi-turn VQA prompts
Q1_PROMPT = "<image>\nExamine the image carefully and identify any visual artifacts. List the artifact regions as bounding boxes in coordinates [x1, y1, x2, y2]."
Q2_PROMPT = "For the region at {bbox_str}, briefly describe what is wrong there. Return a short sentence only."
Q3_PROMPT = "Finally, write a concise description of the image and the anomalies you observed."

def create_dataset_entry(img_path: str, bboxes: List[List[float]], artifacts_data: List[Dict], 
                        caption: str = "", image_width: int = None, image_height: int = None) -> Dict:
    """
    Create a multi-turn dataset entry for localization and explanation (positive data only).
    
    Args:
        img_path: Path to the artifact image
        bboxes: List of bounding boxes [x_min, y_min, x_max, y_max] in pixel coordinates
        artifacts_data: List of artifact metadata dictionaries
        caption: Image caption
        image_width: Image width (optional)
        image_height: Image height (optional)
    
    Returns:
        Dictionary containing the dataset entry
    """
    try:
        conversations = []
        
        # Q1: Detection + Bounding boxes
        conversations.append({
            "from": "human",
            "value": Q1_PROMPT
        })
        
        # Create bbox array with pixel coordinates
        bbox_list = []
        explanations = []
        
        if bboxes and artifacts_data:
            for i, artifact in enumerate(artifacts_data):
                if i < len(bboxes) and len(bboxes[i]) == 4:
                    bbox = bboxes[i]
                    # Use pixel coordinates directly
                    int_bbox = [int(round(coord)) for coord in bbox]
                    bbox_list.append(int_bbox)
                    
                    # Get explanation from artifact data
                    explanation = artifact.get("explanation", artifact.get("label", "Visual artifact detected"))
                    explanations.append(explanation)
        
        # Q1 Response: Just the bbox coordinates
        if bbox_list:
            bbox_json = json.dumps(bbox_list, indent=2)
            conversations.append({
                "from": "gpt",
                "value": f"```json\n{bbox_json}\n```"
            })
            
            # Q2: Individual explanations for each bbox
            for i, (bbox, explanation) in enumerate(zip(bbox_list, explanations)):
                bbox_str = json.dumps(bbox)
                conversations.append({
                    "from": "human",
                    "value": Q2_PROMPT.format(bbox_str=bbox_str)
                })
                conversations.append({
                    "from": "gpt",
                    "value": explanation
                })
            
            # Q3: Overall scene description
            conversations.append({
                "from": "human",
                "value": Q3_PROMPT
            })
            conversations.append({
                "from": "gpt",
                "value": f"```json\n{json.dumps({'explanation': caption}, indent=2)}\n```"
            })
        else:
            # No valid bboxes - provide empty response
            conversations.append({
                "from": "gpt",
                "value": "```json\n[]\n```"
            })
            conversations.append({
                "from": "human",
                "value": Q3_PROMPT
            })
            conversations.append({
                "from": "gpt",
                "value": f"```json\n{json.dumps({'explanation': 'No artifacts detected in this image.'}, indent=2)}\n```"
            })
        
        entry = {
            "images": [img_path],
            "conversations": conversations,
            "has_artifacts": True,
            "artifact_count": len(bbox_list)
        }
        
        # Add image dimensions if available
        if image_width and image_height:
            entry["image_dimensions"] = {"width": image_width, "height": image_height}
        
        return entry
        
    except Exception as e:
        print(f"    ⚠️  Error creating multi-turn dataset entry: {e}")
        raise

def safe_process_artifact_image(image_path: str, metadata_path: str, dataset: List[Dict], 
                              stats: Dict, image_width: int = None, image_height: int = None) -> bool:
    """
    Safely process an artifact image for multi-turn localization training.
    
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
            entry = create_dataset_entry(
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
    Process artifact image sources for multi-turn localization training.
    
    Args:
        artifact_sources: List of source dictionaries with 'path' and 'name' keys
        dataset: List to append processed entries to
        stats: Statistics dictionary to update
        max_samples_per_source: Maximum samples to process per source
    
    Returns:
        Number of samples successfully added
    """
    print(f"🔄 Processing Artifact Sources for Multi-Turn Localization Training")
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
        
        print(f"  ✅ Added {final_added} multi-turn localization samples from {source['name']}")
    
    print(f"✅ Artifact sources processed: {total_added} samples")
    return total_added

def main(target_dataset_size: int = 1000, output_name: str = None):
    """
    Generate multi-turn localization + explanation training dataset (positive data only).
    
    Args:
        target_dataset_size: Target number of positive samples to generate
        output_name: Custom output filename (without extension)
    """
    
    print("🎯 Multi-Turn VQA Localization + Explanation Dataset Generator (Positive Data Only)")
    print("=" * 70)
    print(f"🎯 Target dataset size: {target_dataset_size} (positive samples only)")
    print("=" * 70)
    
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
        output_file = os.path.join(JSON_DIR, f"vqa_loc_exp_multi_turn_{target_dataset_size}.json")
    
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
    print(f"\n📋 Sample Multi-Turn Dataset Entries:")
    print("=" * 70)
    for i, entry in enumerate(dataset[:2]):
        print(f"Sample {i+1}:")
        print(f"  Image: {entry['images'][0]}")
        print(f"  Artifact count: {entry.get('artifact_count', 'N/A')}")
        print(f"  Conversation turns: {len(entry['conversations'])}")
        print(f"  Q1 Response preview: {entry['conversations'][1]['value'][:100]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-turn localization + explanation training dataset (positive data only)")
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
        help="Custom output filename (without extension). If not provided, uses 'vqa_loc_exp_multi_turn_{target_size}'"
    )
    
    args = parser.parse_args()
    
    main(target_dataset_size=args.target_size, output_name=args.output_name)
