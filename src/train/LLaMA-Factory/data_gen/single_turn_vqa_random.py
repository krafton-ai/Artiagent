"""
Corruption-Safe Data JSON Generator

This version uses the robust augmentation module and includes comprehensive
validation and error handling to prevent image corruption and incomplete saves.
"""

import json
import os
import math
import argparse
from PIL import Image
from bbox_aware_augmentations import BBoxAwareAugmentations, robust_validate_bboxes
import random

DATA_DIR = "/data2/jhpark/image-artifacts/data/"
JSON_DIR = "/home/jhpark/image-artifacts/src/train/LLaMA-Factory/data"

def create_dataset_entry(img_path, bboxes, artifacts_data, caption="", is_augmented=False, aug_metadata=None, is_negative_sample=False, image_width=None, image_height=None):
    """Helper to create dataset entry with validation."""
    try:
        if is_negative_sample:
            gpt_response = """{
  "artifact_present": "no",
  "bboxes": [],
  "explanations": [],
  "caption": "There are no artifacts in this image"
}"""
        else:
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
                        norm_x1 = max(0, min(1, norm_x1))
                        norm_y1 = max(0, min(1, norm_y1))
                        norm_x2 = max(0, min(1, norm_x2))
                        norm_y2 = max(0, min(1, norm_y2))
                        
                        normalized_bboxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
                        
                        # Get explanation from artifact data
                        explanation = artifact.get("explanation", artifact.get("label", "Visual artifact detected"))
                        explanations.append(explanation)
            
            if normalized_bboxes and explanations:
                # Format bboxes as JSON arrays
                bbox_str = ",\n    ".join([f"[{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}]" for bbox in normalized_bboxes])
                
                # Format explanations as JSON strings
                explanation_str = ",\n    ".join([f'"{exp}"' for exp in explanations])
                
                gpt_response = f"""{{
                    "artifact_present": "yes",
                    "bboxes": [
                        {bbox_str}
                    ],
                    "explanations": [
                        {explanation_str}
                    ],
                    "caption": "{caption}"
                    }}"""
            else:
                gpt_response = f"""{{
                    "artifact_present": "no",
                    "bboxes": [],
                    "explanations": [],
                    "caption": "{caption}"
                }}"""
        
        entry = {
            "images": [img_path],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nAnalyze this image carefully.\nDescribe whether it contains any visual artifacts,\nwhere those artifacts appear (as bounding boxes normalized to [0,1]),\nand provide short explanations for each localized artifact.\nAlso include a concise caption describing the overall scene.\n\nReturn the results as a valid JSON object with the following keys:\n- artifact_present: \"yes\" or \"no\"\n- bboxes: array of [x1, y1, x2, y2] coordinates normalized to [0,1]\n- explanations: array of strings describing each artifact\n- caption: string describing the overall scene\n\nGenerate your response strictly in English only."
                },
                {
                    "from": "gpt",
                    "value": gpt_response
                }
            ],
            "augmented": is_augmented,
            "negative_sample": is_negative_sample
        }
        
        if is_augmented and aug_metadata:
            entry["augmentation_metadata"] = aug_metadata
        
        return entry
        
    except Exception as e:
        print(f"    ⚠️  Error creating dataset entry: {e}")
        raise

def safe_process_and_augment_image(image_path, bboxes, artifacts, caption, augmentor, dataset, dir_path, base_name, create_dataset_entry, is_negative_sample=False):
    """
    Safely process and augment a single image with comprehensive error handling.
    """
    try:
        # Validate input file
        if not os.path.exists(image_path):
            print(f"  ⚠️  Input image does not exist: {image_path}")
            return False
        
        if not os.access(image_path, os.R_OK):
            print(f"  ⚠️  Cannot read input image: {image_path}")
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
                # Validate image can be loaded and has valid properties
                img.load()  # Force loading to catch corruption early
                
                if img.size[0] < 16 or img.size[1] < 16:
                    print(f"  ⚠️  Image too small {img.size}: {image_path}")
                    return False
                
                if img.size[0] > 4096 or img.size[1] > 4096:
                    print(f"  ⚠️  Image too large {img.size}: {image_path}")
                    return False
                
                img = img.convert('RGB')
                
        except Exception as e:
            print(f"  ⚠️  Cannot load image {image_path}: {e}")
            return False
        
        # Apply augmentations with robust error handling
        aug_image, aug_bboxes, aug_metadata = augmentor.augment_image_and_bboxes(img, bboxes)
        
        # Check if augmentation was successful
        if not aug_metadata.get('success', False):
            print(f"  ⚠️  Augmentation failed: {aug_metadata.get('error', 'Unknown error')}")
            return False
        
        # Validate bboxes for positive samples
        if not is_negative_sample and len(bboxes) > 0:
            aug_bboxes = robust_validate_bboxes(
                aug_bboxes, 
                aug_image.width, 
                aug_image.height,
                min_area=50
            )
            
            if len(aug_bboxes) == 0:
                print(f"  ⚠️  No valid bboxes after augmentation")
                return False
            
            # Create corresponding artifacts data
            valid_artifacts = []
            for i, bbox in enumerate(aug_bboxes):
                if i < len(artifacts):
                    valid_artifacts.append(artifacts[i])
        else:
            valid_artifacts = []
            aug_bboxes = []
        
        # Generate safe filename
        aug_image_name = f"{base_name}_aug.png"
        aug_image_path = os.path.join(dir_path, aug_image_name)
        
        # Ensure output directory exists and is writable
        os.makedirs(dir_path, exist_ok=True)
        if not os.access(dir_path, os.W_OK):
            print(f"  ⚠️  Cannot write to directory: {dir_path}")
            return False
        
        # Use atomic save to prevent corruption
        if not augmentor.safe_save_image(aug_image, aug_image_path):
            print(f"  ⚠️  Failed to save augmented image")
            return False
        
        # Verify saved file
        try:
            with Image.open(aug_image_path) as verify_img:
                if verify_img.size != aug_image.size:
                    print(f"  ⚠️  Saved image size mismatch")
                    os.unlink(aug_image_path)
                    return False
        except Exception as e:
            print(f"  ⚠️  Cannot verify saved image: {e}")
            if os.path.exists(aug_image_path):
                os.unlink(aug_image_path)
            return False
        
        # Add to dataset
        try:
            dataset.append(create_dataset_entry(
                aug_image_path, 
                aug_bboxes, 
                valid_artifacts,
                caption,
                is_augmented=True,
                aug_metadata=aug_metadata,
                is_negative_sample=is_negative_sample,
                image_width=aug_image.width,
                image_height=aug_image.height
            ))
        except Exception as e:
            print(f"  ⚠️  Failed to create dataset entry: {e}")
            if os.path.exists(aug_image_path):
                os.unlink(aug_image_path)
            return False
        
        sample_type = "negative" if is_negative_sample else "positive"
        print(f"  ✅ Generated augmented {sample_type} image: {aug_image_path}")
        if not is_negative_sample:
            print(f"     Original bboxes: {len(bboxes)}, Augmented bboxes: {len(aug_bboxes)}")
        
        return True
    
    except Exception as e:
        sample_type = "negative" if is_negative_sample else "positive"
        print(f"  ⚠️  Error processing {sample_type} image: {e}")
        return False


def safe_convert_to_sft_format(data_dir, dataset, stats, max_samples=None, enable_augmentation=True, augmentation_prob=0.3, is_negative_sample=False):
    """
    Safely convert artifact detection data with comprehensive validation.
    """
    print(f"🔄 Processing directory: {data_dir}")
    
    # Validate input directory
    if not os.path.exists(data_dir):
        print(f"  ❌ Directory does not exist: {data_dir}")
        return
    
    if not os.access(data_dir, os.R_OK):
        print(f"  ❌ Cannot read directory: {data_dir}")
        return
    
    # Initialize robust augmentations
    if enable_augmentation:
        augmentor = BBoxAwareAugmentations(
            augmentation_prob=augmentation_prob,
            resize_range=(0.8, 1.2),
            zoom_range=(0.8, 1.2),
            color_jitter_prob=0.5,
            grayscale_prob=0.1,
            max_image_size=4096,  # Limit to prevent memory issues
            min_crop_size=64      # Prevent tiny crops
        )
        print(f"  ✅ Robust augmentation enabled ({augmentation_prob*100}% probability)")
    else:
        augmentor = None
        print("  ➖ Augmentation disabled")
    
    samples_added = 0 
    processed_count = 0
    corruption_count = 0

    try:
        for root, dirs, files in os.walk(data_dir):
            print(f"    📁 Scanning: {root}")
            for i, img_id in enumerate(dirs):
                if max_samples and samples_added >= max_samples:
                    print(f"    🎯 Reached max samples limit ({max_samples})")
                    break
                
                dir_path = os.path.join(root, img_id)
                metadata_path = os.path.join(dir_path, "metadata.json")
                artifact_image_path = os.path.join(dir_path, "artifact_image.png")
                real_image_path = os.path.join(dir_path, "real_image.png")
                # Check file existence
                has_artifact_image = os.path.exists(artifact_image_path) and os.path.exists(metadata_path)
                has_real_image = os.path.exists(real_image_path)
                if not has_artifact_image and not has_real_image:
                    continue
                
                try:
                    # Load metadata safely
                    artifacts = []
                    caption = "There are no artifacts in this image. """
                    original_bboxes = []
                    if has_artifact_image:
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                            artifacts = metadata.get("artifacts", [])
                            caption = metadata.get("caption", "An image with artifacts")
                            # Extract and validate bboxes
                            for artifact in artifacts:
                                bbox = artifact.get("target_bbox")
                                if bbox and len(bbox) == 4:
                                    try:
                                        bbox = [float(coord) for coord in bbox]
                                        if all(math.isfinite(coord) for coord in bbox):
                                            original_bboxes.append(bbox)
                                    except (ValueError, TypeError):
                                        continue
                        except Exception as e:
                            print(f"    ⚠️  Error loading metadata for {img_id}: {e}")
                            continue
                        
                    # Process artifact image
                    if has_artifact_image:
                        if augmentor and augmentor.should_augment():
                            success = safe_process_and_augment_image(
                                artifact_image_path, original_bboxes, artifacts, caption,
                                augmentor, dataset, dir_path, "artifact_image", create_dataset_entry, is_negative_sample
                            )
                            if success:
                                stats['augmented_count'] += 1
                                samples_added += 1
                            else:
                                corruption_count += 1
                        else:
                            # Add original image with validation
                            try:
                                # Get image dimensions for bbox normalization
                                with Image.open(artifact_image_path) as img:
                                    img_width, img_height = img.size
                                
                                dataset.append(create_dataset_entry(
                                    artifact_image_path, original_bboxes, artifacts, caption,
                                    image_width=img_width, image_height=img_height
                                ))
                                stats['original_count'] += 1
                                samples_added += 1
                            except Exception as e:
                                print(f"    ⚠️  Error adding original artifact image: {e}")
                                corruption_count += 1
                    processed_count += 1
                    stats['processed_count'] += 1
                    if samples_added % 50 == 0 and samples_added > 0:
                        print(f"    📊 Progress: {processed_count} processed, {corruption_count} corrupted/failed")
                except Exception as e:
                    print(f"    ⚠️  Error processing {img_id}: {e}")
                    corruption_count += 1
                    continue
                
    except Exception as e:
        print(f"  ❌ Error walking directory: {e}")
        return
    
    # Print statistics
    print(f"  ✅ Completed: {processed_count} processed")
    if corruption_count > 0:
        print(f"  ⚠️  Corrupted/failed: {corruption_count}")
    
    # Show augmentation statistics if enabled
    if augmentor:
        aug_stats = augmentor.get_statistics()
        print(f"  📈 Augmentation stats:")
        print(f"      Success rate: {aug_stats['success_rate']:.1%}")
        print(f"      Zoom failures: {aug_stats['zoom_failure_rate']:.1%}")
        print(f"      Save failures: {aug_stats['save_failure_rate']:.1%}")

def process_negative_images(data_dir, dataset, stats, max_samples=None, enable_augmentation=True, augmentation_prob=0.3):
    """
    Process negative images from a directory of image files.
    """
    print(f"🔄 Processing negative images from: {data_dir}")
    
    # Validate input directory
    if not os.path.exists(data_dir):
        print(f"  ❌ Directory does not exist: {data_dir}")
        return
    
    if not os.access(data_dir, os.R_OK):
        print(f"  ❌ Cannot read directory: {data_dir}")
        return
    
    # Initialize augmentations
    if enable_augmentation:
        augmentor = BBoxAwareAugmentations(
            augmentation_prob=augmentation_prob,
            resize_range=(0.8, 1.2),
            zoom_range=(0.8, 1.2),
            color_jitter_prob=0.5,
            grayscale_prob=0.1,
            max_image_size=4096,
            min_crop_size=64
        )
        print(f"  ✅ Augmentation enabled for negatives ({augmentation_prob*100}% probability)")
    else:
        augmentor = None
        print("  ➖ Augmentation disabled for negatives")
    
    # Get all image files
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))
    
    # Shuffle and limit
    random.shuffle(image_files)
    if max_samples:
        image_files = image_files[:max_samples]
    
    samples_added = 0
    processed_count = 0
    corruption_count = 0
    
    print(f"  📊 Processing {len(image_files)} negative images...")
    
    for image_path in image_files:
        if max_samples and samples_added >= max_samples:
            print(f"    🎯 Reached target negative samples ({max_samples}) - stopping")
            break
        try:
            # Load and validate image
            try:
                with Image.open(image_path) as img:
                    img.load()
                    if img.size[0] < 16 or img.size[1] < 16:
                        continue
                    img = img.convert('RGB')
                    
            except Exception as e:
                print(f"  ⚠️  Cannot load image {image_path}: {e}")
                corruption_count += 1
                continue
            
            # Process with or without augmentation
            if augmentor and augmentor.should_augment():
                # Apply augmentation
                aug_image, _, aug_metadata = augmentor.augment_image_and_bboxes(img, [])
                
                if not aug_metadata.get('success', False):
                    corruption_count += 1
                    continue
                
                # Generate augmented filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                aug_path = os.path.join(os.path.dirname(image_path), f"{base_name}_neg_aug.png")
                
                # Save augmented image
                if not augmentor.safe_save_image(aug_image, aug_path):
                    corruption_count += 1
                    continue
                
                # Add to dataset
                try:
                    dataset.append(create_dataset_entry(
                        aug_path, [], [], "There are no artifacts in this image.",
                        is_augmented=True, aug_metadata=aug_metadata, is_negative_sample=True,
                        image_width=aug_image.width, image_height=aug_image.height
                    ))
                    stats['augmented_count'] += 1
                    samples_added += 1
                except Exception as e:
                    print(f"  ⚠️  Error creating augmented negative entry: {e}")
                    corruption_count += 1
                    continue
            else:
                # Use original image
                try:
                    dataset.append(create_dataset_entry(
                        image_path, [], [], "There are no artifacts in this image.",
                        is_augmented=False, is_negative_sample=True,
                        image_width=img.width, image_height=img.height
                    ))
                    stats['original_count'] += 1
                    samples_added += 1
                except Exception as e:
                    print(f"  ⚠️  Error creating original negative entry: {e}")
                    corruption_count += 1
                    continue
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"    📊 Progress: {processed_count} negative images processed")
        
        except Exception as e:
            print(f"  ⚠️  Error processing negative image {image_path}: {e}")
            corruption_count += 1
            continue
    
    print(f"  ✅ Processed {processed_count} negative images")
    if corruption_count > 0:
        print(f"  ⚠️  Failed: {corruption_count} negative images")

def main(target_dataset_size=2000, output_name=None):
    """Generate corruption-safe balanced dataset.
    
    Args:
        target_dataset_size (int): Total number of samples to generate.
                                  Will be split equally between artifact and negative samples.
        output_name (str): Custom output filename (without extension).
    """
    
    print("🛡️  Corruption-Safe Balanced Dataset Generator")
    print("=" * 50)
    print(f"🎯 Target dataset size: {target_dataset_size}")
    
    # Calculate balanced sample counts
    artifact_samples = target_dataset_size // 2
    negative_samples = target_dataset_size - artifact_samples
    
    print(f"📊 Balanced distribution:")
    print(f"   Artifact samples: {artifact_samples}")
    print(f"   Negative samples: {negative_samples}")
    print("=" * 50)
    
    dataset = []
    stats = {'processed_count': 0, 'augmented_count': 0, 'original_count': 0}

    # data sources (TODO: UPDATE THESE PATHS)
    artifact_sources = [
        # {
        #     'path': "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/data/artifacts/filtered_animals_1k_fireflow_25", 
        #     'max_samples': 250,
        #     'name': "coco_animal_fireflow_same"
        # },
        # {
        #     'path': "/home/jovyan/image-artifacts/data/filtered_data_synth/coco/animal", 
        #     'max_samples': None,
        #     'name': "coco_animal_fireflow"
        # },
        # {
        #     'path': "/home/jovyan/image-artifacts/data/filtered_data_synth/coco/person", 
        #     'max_samples': 250,
        #     'name': "coco_person_fireflow"
        # },
        {
            'path': os.path.join(DATA_DIR, "train/vanilla/filtered_animals_1k"),
            'max_samples': None,  # Will be set dynamically
            'name': "coco_animal_vanilla"
        },
        {
            'path': os.path.join(DATA_DIR, "train/vanilla/filtered_person_1k"),
            'max_samples': None,  # Will be set dynamically
            'name': "coco_person_vanilla"
        }
    ]
    
    # Distribute artifact samples across available sources
    available_artifact_sources = [s for s in artifact_sources if os.path.exists(s['path'])]
    if available_artifact_sources:
        samples_per_source = artifact_samples // len(available_artifact_sources)
        remainder = artifact_samples % len(available_artifact_sources)
        
        for i, source in enumerate(available_artifact_sources):
            source['max_samples'] = samples_per_source + (1 if i < remainder else 0)
            print(f"   {source['name']}: {source['max_samples']} samples")
    else:
        print("⚠️  No artifact sources available!")
        return
    negative_sources = [
        {
            'path': os.path.join(DATA_DIR, "sources/coco/train2017"), 
            'max_samples': None,  # Will be set dynamically
            'name': "coco"
        },
        # {
        #     'path': "/home/jovyan/image-artifacts/data/image-artifact-real-images/caltech", 
        #     'max_samples': None,
        #     'name': "caltech"
        # },
        # {
        #     'path': "/home/jovyan/image-artifacts/data/image-artifact-real-images/celebahq", 
        #     'max_samples': None,
        #     'name': "celebahq"
        # },
        # {
        #     'path':  "/home/jovyan/image-artifacts/data/image-artifact-real-images/hands", 
        #     'max_samples': None,
        #     'name': "hands"
        # }
    ]
    
    # Distribute negative samples across available sources
    available_negative_sources = [s for s in negative_sources if os.path.exists(s['path'])]
    if available_negative_sources:
        samples_per_source = negative_samples // len(available_negative_sources)
        remainder = negative_samples % len(available_negative_sources)
        
        for i, source in enumerate(available_negative_sources):
            source['max_samples'] = samples_per_source + (1 if i < remainder else 0)
            print(f"   {source['name']}: {source['max_samples']} samples")
    else:
        print("⚠️  No negative sources available!")
        return
    
    # Process artifact sources
    print(f"\n📊 Processing Artifact Sources")
    print("=" * 40)
    artifact_count = 0
    for source in available_artifact_sources:
        print(f"\n🔄 Processing {source['name']}")
        initial_dataset_size = len(dataset)
        safe_convert_to_sft_format(
            source['path'], dataset, stats, source['max_samples'],
            enable_augmentation=False, augmentation_prob=0.3
        )
        added_samples = len(dataset) - initial_dataset_size
        artifact_count += added_samples
        print(f"  ✅ Added {added_samples} artifact samples from {source['name']}")
    
    print(f"✅ Artifact sources processed: {artifact_count} samples")
    print(f"  Total dataset entries: {len(dataset)}")

    # Process negative sources  
    print(f"\n📊 Processing Negative Sources")
    print("=" * 40)
    negative_count = 0
    for source in available_negative_sources:
        print(f"\n🔄 Processing {source['name']}")
        initial_dataset_size = len(dataset)
        process_negative_images(
            source['path'], dataset, stats, source['max_samples'],
            enable_augmentation=False, augmentation_prob=0.3
        )
        added_samples = len(dataset) - initial_dataset_size
        negative_count += added_samples
        print(f"  ✅ Added {added_samples} negative samples from {source['name']}")
    
    print(f"✅ Negative sources processed: {negative_count} samples")
    print(f"  Total dataset entries: {len(dataset)}")

    # Validate balanced dataset
    print(f"\n🔍 Dataset Balance Validation:")
    print("=" * 40)
    actual_artifact_count = sum(1 for entry in dataset if not entry.get('negative_sample', False))
    actual_negative_count = sum(1 for entry in dataset if entry.get('negative_sample', False))
    
    print(f"  Target artifact samples: {artifact_samples}")
    print(f"  Actual artifact samples: {actual_artifact_count}")
    print(f"  Target negative samples: {negative_samples}")
    print(f"  Actual negative samples: {actual_negative_count}")
    print(f"  Balance ratio: {actual_artifact_count}:{actual_negative_count}")
    
    if actual_artifact_count == 0 or actual_negative_count == 0:
        print("⚠️  Warning: Dataset is not balanced - one class has no samples!")
    elif abs(actual_artifact_count - actual_negative_count) > 1:
        print("⚠️  Warning: Dataset is not perfectly balanced!")
    else:
        print("✅ Dataset is well balanced!")
    
    # Shuffle the final dataset
    print(f"\n🔀 Shuffling final dataset...")
    random.shuffle(dataset)
    
    # Save with validation
    if output_name:
        output_file = os.path.join(JSON_DIR, f"{output_name}.json")
    else:
        output_file = os.path.join(JSON_DIR, f"artifact_balanced_{target_dataset_size}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Dataset saved: {output_file}")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"  Target dataset size: {target_dataset_size}")
    print(f"  Actual dataset size: {len(dataset)}")
    print(f"  Artifact samples: {actual_artifact_count}")
    print(f"  Negative samples: {actual_negative_count}")
    print(f"  Augmented samples: {stats['augmented_count']}")
    print(f"  Original samples: {stats['original_count']}")
    print(f"  Balance achieved: {'Yes' if abs(actual_artifact_count - actual_negative_count) <= 1 else 'No'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced artifact detection dataset")
    parser.add_argument(
        "--target_size", 
        type=int, 
        default=2000,
        help="Target total dataset size (will be split equally between artifact and negative samples)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom output filename (without extension). If not provided, uses 'artifact_balanced_{target_size}'"
    )
    
    args = parser.parse_args()
    
    main(target_dataset_size=args.target_size, output_name=args.output_name)
