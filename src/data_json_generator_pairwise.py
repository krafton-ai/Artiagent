"""
Corruption-Safe Data JSON Generator

This version uses the robust augmentation module and includes comprehensive
validation and error handling to prevent image corruption and incomplete saves.
"""

import json
import os
import math
from PIL import Image
from bbox_aware_augmentations import BBoxAwareAugmentations, robust_validate_bboxes
import random

def create_dataset_entry(img_path, bboxes, artifacts_data, caption="", is_augmented=False, aug_metadata=None, is_negative_sample=False):
    """Helper to create dataset entry with validation."""
    try:
        bbox_label_pairs = []
        
        if is_negative_sample:
            gpt_response = "There are no artifacts in this image. [] ```json\{'has_artifacts': false\}```"
        else:
            for i, artifact in enumerate(artifacts_data):
                if i < len(bboxes) and len(bboxes[i]) == 4:
                    bbox = bboxes[i]
                    label = artifact.get("label", "Unknown")
                    bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
                    bbox_label_pairs.append(f"{label}: {bbox_str}")
            
            if bbox_label_pairs:
                gpt_response = f"{caption} [{', '.join(bbox_label_pairs)}] " + "```json\{'has_artifacts': true\}```"
            else:
                gpt_response = f"{caption} "  + "```json\{'has_artifacts': true\}```"
        
        entry = {
            "images": [img_path],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>Analyze the image and describe any visual anomalies. Provide bounding boxes and explain in detail."
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
                is_negative_sample=is_negative_sample
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


def safe_convert_to_sft_format(data_dir, dataset, stats, samples=None, enable_augmentation=True, augmentation_prob=0.3, is_negative_sample=False):
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
                if samples and samples_added >= samples:
                    print(f"    🎯 Reached samples limit ({samples})")
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
                        
                        # Process images as pairs - both artifact and real images
                        pair_added = False
                        
                        if has_artifact_image and has_real_image:
                            # Both images exist - process as positive-negative pair
                            artifact_success = False
                            real_success = False
                            
                            # Process artifact image (positive sample)
                            if augmentor and augmentor.should_augment():
                                artifact_success = safe_process_and_augment_image(
                                    artifact_image_path, original_bboxes, artifacts, caption,
                                    augmentor, dataset, dir_path, "artifact_image", create_dataset_entry, False
                                )
                                if not artifact_success:
                                    # Fallback to original artifact image
                                    try:
                                        dataset.append(create_dataset_entry(artifact_image_path, original_bboxes, artifacts, caption))
                                        stats['original_count'] += 1
                                        artifact_success = True
                                    except Exception as e:
                                        print(f"    ⚠️  Error adding original artifact image: {e}")
                                else:
                                    stats['augmented_count'] += 1
                            else:
                                try:
                                    dataset.append(create_dataset_entry(artifact_image_path, original_bboxes, artifacts, caption))
                                    stats['original_count'] += 1
                                    artifact_success = True
                                except Exception as e:
                                    print(f"    ⚠️  Error adding original artifact image: {e}")
                            
                            # Process real image (negative sample)
                            if augmentor and augmentor.should_augment():
                                real_success = safe_process_and_augment_image(
                                    real_image_path, [], [], "There are no artifacts in this image.",
                                    augmentor, dataset, dir_path, "real_image", create_dataset_entry, True
                                )
                                if not real_success:
                                    # Fallback to original real image
                                    try:
                                        dataset.append(create_dataset_entry(real_image_path, [], [], "There are no artifacts in this image.", is_negative_sample=True))
                                        stats['original_count'] += 1
                                        real_success = True
                                    except Exception as e:
                                        print(f"    ⚠️  Error adding original real image: {e}")
                                else:
                                    stats['augmented_count'] += 1
                            else:
                                try:
                                    dataset.append(create_dataset_entry(real_image_path, [], [], "There are no artifacts in this image.", is_negative_sample=True))
                                    stats['original_count'] += 1
                                    real_success = True
                                except Exception as e:
                                    print(f"    ⚠️  Error adding original real image: {e}")
                            
                            # Count as pair only if both succeeded
                            if artifact_success and real_success:
                                samples_added += 1  # Count as one pair
                                pair_added = True
                                print(f"  ✅ Added pair: artifact + real image for {img_id}")
                            else:
                                corruption_count += 1
                                print(f"  ⚠️  Failed to add complete pair for {img_id}")
                        
                        elif has_artifact_image:
                            continue
                            
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

def main():
    """Generate corruption-safe dataset."""
    
    print("🛡️  Corruption-Safe Dataset Generator")
    print("=" * 50)
    
    dataset = []
    stats = {'processed_count': 0, 'augmented_count': 0, 'original_count': 0}

    # data sources (TODO : UPDATE THESE PATHS)
    artifact_sources = [    
        # {
        #     'path': "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/data/artifacts/filtered_animals_1k_fireflow_25", 
        #     'samples': 250,
        #     'name': "coco_animal_fireflow_same"
        # },
        # {
        #     'path': "/home/jovyan/image-artifacts/data/filtered_data_synth/coco/animal", 
        #     'samples': 250,
        #     'name': "coco_animal_fireflow"
        # },
        # {
        #     'path': "/home/jovyan/image-artifacts/data/filtered_data_synth/coco/person", 
        #     'samples': 250,
        #     'name': "coco_person_fireflow"
        # },
        {
            'path': "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/data/artifacts/filtered_animals_1k", 
            'samples': 250,
            'name': "coco_animal_vanilla"
        },
        {
            'path': "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/data/artifacts/filtered_person_1k", 
            'samples': 250,
            'name': "coco_person_vanilla"
        }
    ]
    
    # Process artifact sources
    print(f"\n📊 Processing Artifact Sources")
    print("=" * 40)
    for source in artifact_sources:
       if os.path.exists(source['path']):
           safe_convert_to_sft_format(
               source['path'], dataset, stats, source['samples'],
               enable_augmentation=True, augmentation_prob=0.3
           )
       else:
           print(f"⚠️  Path does not exist: {source['path']}")
    
    print(f"✅ Artifact sources processed: {stats['processed_count']} samples")
    print(f"  Total dataset entries: {len(dataset)}")


    # Shuffle the final dataset
    print(f"\n🔀 Shuffling final dataset...")
    groups = [dataset[i:i+2] for i in range(0, len(dataset), 2)]
    random.shuffle(groups)
    shuffled_dataset = [item for group in groups for item in group]
    
    # Save with validation
    output_file = "artifact_1k_fireflow.json"       # TODO : designate output path
    try:
        with open(output_file, 'w') as f:
            json.dump(shuffled_dataset, f, indent=2)
        
        print(f"✅ Dataset saved: {output_file}")
                 
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"  Total dataset size: {len(dataset)}")
    print(f"  Successful processing rate: {stats['processed_count']}")
    print(f"  Augmented samples: {stats['augmented_count']}")
    print(f"  Original samples: {stats['original_count']}")


if __name__ == "__main__":
    main()
