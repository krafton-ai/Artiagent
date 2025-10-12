"""
Robust BBox-Aware Image Augmentation Module with Corruption Prevention

This module provides image augmentations with comprehensive validation and error handling
to prevent image corruption, incomplete saves, and invalid transformations.
"""

import random
import math
import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import colorsys
import tempfile
import shutil


class BBoxAwareAugmentations:
    """
    Robust image augmentation class that prevents corruption and validates all operations.
    
    Key safety features:
    - Comprehensive coordinate validation
    - Atomic file operations
    - Memory usage monitoring
    - Detailed error logging
    - Image integrity verification
    """
    
    def __init__(
        self,
        augmentation_prob: float = 0.3,
        resize_range: Tuple[float, float] = (0.8, 1.2),  # ±20%
        zoom_range: Tuple[float, float] = (0.8, 1.2),    # ±20%
        color_jitter_prob: float = 0.5,
        grayscale_prob: float = 0.1,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        min_bbox_area_ratio: float = 0.1,
        max_image_size: int = 4096,  # Prevent memory issues with huge images
        min_crop_size: int = 32,     # Minimum crop size to prevent tiny crops
    ):
        """Initialize robust augmentations with safety limits."""
        self.augmentation_prob = augmentation_prob
        self.resize_range = resize_range
        self.zoom_range = zoom_range
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.max_image_size = max_image_size
        self.min_crop_size = min_crop_size
        
        # Statistics for debugging
        self.stats = {
            'total_attempts': 0,
            'successful_augmentations': 0,
            'zoom_failures': 0,
            'resize_failures': 0,
            'save_failures': 0,
            'validation_failures': 0
        }
    
    def _validate_image(self, image: Image.Image, operation_name: str) -> bool:
        """Validate image integrity and constraints."""
        try:
            # Check basic image properties
            if image is None:
                print(f"  ⚠️  {operation_name}: Image is None")
                return False
            
            if not hasattr(image, 'size') or not image.size:
                print(f"  ⚠️  {operation_name}: Invalid image size")
                return False
            
            width, height = image.size
            
            # Check size constraints
            if width <= 0 or height <= 0:
                print(f"  ⚠️  {operation_name}: Invalid dimensions {width}x{height}")
                return False
            
            if width > self.max_image_size or height > self.max_image_size:
                print(f"  ⚠️  {operation_name}: Image too large {width}x{height} > {self.max_image_size}")
                return False
            
            if width < 16 or height < 16:
                print(f"  ⚠️  {operation_name}: Image too small {width}x{height}")
                return False
            
            # Try to access image data to ensure it's not corrupted
            try:
                image.load()
            except Exception as e:
                print(f"  ⚠️  {operation_name}: Cannot load image data: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  {operation_name}: Image validation error: {e}")
            return False
    
    def _validate_crop_region(self, image: Image.Image, crop_box: Tuple[int, int, int, int]) -> bool:
        """Validate crop region to prevent PIL errors."""
        try:
            width, height = image.size
            x1, y1, x2, y2 = crop_box
            
            # Check bounds
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                print(f"  ⚠️  Crop region {crop_box} exceeds image bounds {width}x{height}")
                return False
            
            # Check order
            if x2 <= x1 or y2 <= y1:
                print(f"  ⚠️  Invalid crop region {crop_box}: x2 <= x1 or y2 <= y1")
                return False
            
            # Check minimum size
            crop_width = x2 - x1
            crop_height = y2 - y1
            if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                print(f"  ⚠️  Crop too small: {crop_width}x{crop_height} < {self.min_crop_size}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Crop validation error: {e}")
            return False
    
    def should_augment(self) -> bool:
        """Check if this image should be augmented."""
        return random.random() < self.augmentation_prob
    
    def resize_image_and_bboxes(
        self, 
        image: Image.Image, 
        bboxes: List[List[float]], 
        scale_factor: Optional[float] = None
    ) -> Tuple[Image.Image, List[List[float]], bool]:
        """
        Safely resize image and scale bounding box coordinates.
        
        Returns:
            Tuple of (resized_image, scaled_bboxes, success_flag)
        """
        try:
            if not self._validate_image(image, "resize_input"):
                self.stats['resize_failures'] += 1
                return image, bboxes, False
            
            if scale_factor is None:
                scale_factor = random.uniform(*self.resize_range)
            
            # Validate scale factor
            if scale_factor <= 0.1 or scale_factor > 5.0:
                print(f"  ⚠️  Invalid scale factor: {scale_factor}")
                self.stats['resize_failures'] += 1
                return image, bboxes, False
            
            orig_width, orig_height = image.size
            
            # Calculate new dimensions with safety limits
            new_width = max(16, min(int(orig_width * scale_factor), self.max_image_size))
            new_height = max(16, min(int(orig_height * scale_factor), self.max_image_size))
            
            # Recalculate actual scale factors (may differ due to clipping)
            actual_scale_x = new_width / orig_width
            actual_scale_y = new_height / orig_height
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if not self._validate_image(resized_image, "resize_output"):
                self.stats['resize_failures'] += 1
                return image, bboxes, False
            
            # Scale bounding boxes with actual scale factors
            scaled_bboxes = []
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                    
                x_min, y_min, x_max, y_max = bbox
                
                # Scale coordinates
                scaled_bbox = [
                    x_min * actual_scale_x,
                    y_min * actual_scale_y,
                    x_max * actual_scale_x,
                    y_max * actual_scale_y
                ]
                
                # Validate scaled bbox
                if (scaled_bbox[2] > scaled_bbox[0] and 
                    scaled_bbox[3] > scaled_bbox[1] and
                    all(0 <= coord <= new_width if i % 2 == 0 else coord <= new_height 
                        for i, coord in enumerate(scaled_bbox))):
                    scaled_bboxes.append(scaled_bbox)
            
            return resized_image, scaled_bboxes, True
            
        except Exception as e:
            print(f"  ⚠️  Resize error: {e}")
            self.stats['resize_failures'] += 1
            return image, bboxes, False
    
    def zoom_image_and_bboxes(
        self, 
        image: Image.Image, 
        bboxes: List[List[float]]
    ) -> Tuple[Image.Image, List[List[float]], bool]:
        """
        Safely zoom (crop + resize) image while ensuring bboxes are preserved.
        
        Returns:
            Tuple of (zoomed_image, adjusted_bboxes, success_flag)
        """
        try:
            if not self._validate_image(image, "zoom_input"):
                self.stats['zoom_failures'] += 1
                return image, bboxes, False
            
            orig_width, orig_height = image.size
            
            # Handle empty bboxes
            if not bboxes:
                zoom_factor = random.uniform(*self.zoom_range)

                # Handle edge case: if zoom factor is too close to 1.0, no effective zoom
                if abs(zoom_factor - 1.0) < 0.01:
                    return image, [], False

                # Calculate random crop position for balanced augmentation
                crop_width = orig_width / zoom_factor
                crop_height = orig_height / zoom_factor

                # Ensure crop doesn't exceed image boundaries
                max_crop_x = max(0, orig_width - crop_width)
                max_crop_y = max(0, orig_height - crop_height)

                crop_x = random.uniform(0, max_crop_x)
                crop_y = random.uniform(0, max_crop_y)
    
                return self._apply_safe_zoom(image, [], zoom_factor, crop_x, crop_y)
            
            # Calculate union of all bboxes with safety margins
            try:
                all_x_mins = [max(0, bbox[0]) for bbox in bboxes if len(bbox) >= 4]
                all_y_mins = [max(0, bbox[1]) for bbox in bboxes if len(bbox) >= 4]  
                all_x_maxs = [min(orig_width, bbox[2]) for bbox in bboxes if len(bbox) >= 4]
                all_y_maxs = [min(orig_height, bbox[3]) for bbox in bboxes if len(bbox) >= 4]
                
                if not all_x_mins:  # No valid bboxes
                    self.stats['zoom_failures'] += 1
                    return image, bboxes, False
                
                union_x_min = min(all_x_mins)
                union_y_min = min(all_y_mins)
                union_x_max = max(all_x_maxs)
                union_y_max = max(all_y_maxs)
                
                # Add safety margins
                bbox_width = union_x_max - union_x_min
                bbox_height = union_y_max - union_y_min
                margin_x = max(5, bbox_width * 0.02)  # 2% margin or 5px minimum
                margin_y = max(5, bbox_height * 0.02)
                
                union_x_min = max(0, union_x_min - margin_x)
                union_y_min = max(0, union_y_min - margin_y)
                union_x_max = min(orig_width, union_x_max + margin_x)
                union_y_max = min(orig_height, union_y_max + margin_y)
                
            except Exception as e:
                print(f"  ⚠️  BBox union calculation error: {e}")
                self.stats['zoom_failures'] += 1
                return image, bboxes, False
            
            # Calculate maximum safe zoom with conservative limits
            try:
                # Calculate max zoom factors for each dimension
                max_zoom_left = orig_width / union_x_max if union_x_max > 0 else 1.0
                max_zoom_top = orig_height / union_y_max if union_y_max > 0 else 1.0
                max_zoom_right = orig_width / (orig_width - union_x_min) if union_x_min < orig_width else 1.0
                max_zoom_bottom = orig_height / (orig_height - union_y_min) if union_y_min < orig_height else 1.0
                
                max_zoom = min(max_zoom_left, max_zoom_top, max_zoom_right, max_zoom_bottom)
                max_zoom = max(1.0, min(max_zoom, 3.0))  # Limit to reasonable range
                
                # Choose conservative zoom factor
                zoom_upper_limit = min(self.zoom_range[1], max_zoom * 0.8)  # 80% of theoretical max
                if zoom_upper_limit <= self.zoom_range[0]:
                    self.stats['zoom_failures'] += 1
                    return image, bboxes, False
                
                zoom_factor = random.uniform(self.zoom_range[0], zoom_upper_limit)
                
            except Exception as e:
                print(f"  ⚠️  Zoom factor calculation error: {e}")
                self.stats['zoom_failures'] += 1
                return image, bboxes, False
            
            # Calculate and validate crop region
            crop_width = orig_width / zoom_factor
            crop_height = orig_height / zoom_factor
            
            if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                self.stats['zoom_failures'] += 1
                return image, bboxes, False
            
            # Calculate safe crop position
            min_crop_x = max(0, union_x_max - crop_width)
            max_crop_x = min(union_x_min, orig_width - crop_width)
            min_crop_y = max(0, union_y_max - crop_height)
            max_crop_y = min(union_y_min, orig_height - crop_height)
            
            if min_crop_x >= max_crop_x or min_crop_y >= max_crop_y:
                self.stats['zoom_failures'] += 1
                return image, bboxes, False
            
            # Choose crop position
            crop_x = random.uniform(min_crop_x, max_crop_x)
            crop_y = random.uniform(min_crop_y, max_crop_y)
            
            return self._apply_safe_zoom(image, bboxes, zoom_factor, crop_x, crop_y)
            
        except Exception as e:
            print(f"  ⚠️  Zoom error: {e}")
            self.stats['zoom_failures'] += 1
            return image, bboxes, False
    
    def _apply_safe_zoom(
        self, 
        image: Image.Image, 
        bboxes: List[List[float]], 
        zoom_factor: float, 
        crop_x: float, 
        crop_y: float
    ) -> Tuple[Image.Image, List[List[float]], bool]:
        """Apply zoom transformation with comprehensive safety checks."""
        try:
            orig_width, orig_height = image.size

            excessive_truncation = False
            
            # Calculate crop dimensions
            crop_width = orig_width / zoom_factor
            crop_height = orig_height / zoom_factor
            
            # Create and validate crop box
            crop_box = (
                max(0, int(crop_x)),
                max(0, int(crop_y)), 
                min(orig_width, int(crop_x + crop_width)),
                min(orig_height, int(crop_y + crop_height))
            )
            
            if not self._validate_crop_region(image, crop_box):
                return image, bboxes, False
            
            # Perform crop
            cropped_image = image.crop(crop_box)
            
            if not self._validate_image(cropped_image, "crop_result"):
                return image, bboxes, False
            
            # Resize back to original size
            zoomed_image = cropped_image.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
            
            if not self._validate_image(zoomed_image, "zoom_result"):
                return image, bboxes, False
            
            # Adjust bounding boxes
            adjusted_bboxes = []
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                    
                x_min, y_min, x_max, y_max = bbox
                
                # Adjust for crop offset and zoom factor
                adj_x_min = (x_min - crop_x) * zoom_factor
                adj_y_min = (y_min - crop_y) * zoom_factor
                adj_x_max = (x_max - crop_x) * zoom_factor
                adj_y_max = (y_max - crop_y) * zoom_factor
                
                # Calculate unclamped bbox area
                unclamped_width = adj_x_max - adj_x_min
                unclamped_height = adj_y_max - adj_y_min
                unclamped_area = unclamped_width * unclamped_height

                # Clamp to image boundaries
                clamped_x_min = max(0, adj_x_min)
                clamped_y_min = max(0, adj_y_min)
                clamped_x_max = min(orig_width, adj_x_max)
                clamped_y_max = min(orig_height, adj_y_max)

                # Calculate clamped bbox area
                clamped_width = clamped_x_max - clamped_x_min
                clamped_height = clamped_y_max - clamped_y_min
                clamped_area = clamped_width * clamped_height

                # Check for excessive truncation
                if unclamped_area > 0:
                    truncation_ratio = (unclamped_area - clamped_area) / unclamped_area
                    if truncation_ratio > 0:
                        excessive_truncation = True
                        break
                
                excessive_truncation = False
                # Check if clamped bbox is still valid (has minimum area)
                orig_area = orig_width * orig_height
                if clamped_area / orig_area >= self.min_bbox_area_ratio and clamped_width > 0 and clamped_height > 0:
                    adjusted_bboxes.append([clamped_x_min, clamped_y_min, clamped_x_max, clamped_y_max])
            
            # Fall back to original image if excessive truncation detected
            if bboxes and excessive_truncation:
                return image, bboxes, False

            return zoomed_image, adjusted_bboxes, True
            
        except Exception as e:
            print(f"  ⚠️  Safe zoom application error: {e}")
            return image, bboxes, False
    
    def apply_color_augmentations(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Apply color augmentations with error handling."""
        try:
            if not self._validate_image(image, "color_input"):
                return image, False
            
            if random.random() > self.color_jitter_prob:
                return image, True
            
            result_image = image.copy()
            
            # Apply transformations safely
            transforms = []
            if random.random() < 0.7:
                transforms.append(('brightness', random.uniform(*self.brightness_range)))
            if random.random() < 0.7:
                transforms.append(('contrast', random.uniform(*self.contrast_range)))
            if random.random() < 0.8:
                transforms.append(('saturation', random.uniform(*self.saturation_range)))
            if random.random() < 0.6:
                transforms.append(('hue', random.uniform(*self.hue_range)))
            
            random.shuffle(transforms)
            
            for transform_type, factor in transforms:
                try:
                    if transform_type == 'brightness':
                        enhancer = ImageEnhance.Brightness(result_image)
                        result_image = enhancer.enhance(factor)
                    elif transform_type == 'contrast':
                        enhancer = ImageEnhance.Contrast(result_image)
                        result_image = enhancer.enhance(factor)
                    elif transform_type == 'saturation':
                        enhancer = ImageEnhance.Color(result_image)
                        result_image = enhancer.enhance(factor)
                    elif transform_type == 'hue':
                        result_image = self._adjust_hue_safe(result_image, factor)
                    
                    if not self._validate_image(result_image, f"color_{transform_type}"):
                        return image, False
                        
                except Exception as e:
                    print(f"  ⚠️  Color transform {transform_type} failed: {e}")
                    return image, False
            
            return result_image, True
            
        except Exception as e:
            print(f"  ⚠️  Color augmentation error: {e}")
            return image, False
    
    def _adjust_hue_safe(self, image: Image.Image, hue_shift: float) -> Image.Image:
        """Safely adjust hue with error handling."""
        try:
            # Convert to HSV using PIL's built-in method when possible
            if abs(hue_shift) < 0.001:  # Skip tiny adjustments
                return image
                
            # Use numpy for hue adjustment with bounds checking
            img_array = np.array(image).astype(np.float32) / 255.0
            
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                return image
                
            hsv_img = np.zeros_like(img_array)
            
            for i in range(min(img_array.shape[0], 1000)):  # Limit for performance
                for j in range(min(img_array.shape[1], 1000)):
                    try:
                        r, g, b = img_array[i, j]
                        h, s, v = colorsys.rgb_to_hsv(r, g, b)
                        h = (h + hue_shift) % 1.0
                        hsv_img[i, j] = colorsys.hsv_to_rgb(h, s, v)
                    except:
                        hsv_img[i, j] = img_array[i, j]  # Keep original on error
            
            # Apply to full image if sample worked
            if hsv_img[0, 0].sum() > 0:
                for i in range(img_array.shape[0]):
                    for j in range(img_array.shape[1]):
                        try:
                            r, g, b = img_array[i, j]
                            h, s, v = colorsys.rgb_to_hsv(r, g, b)
                            h = (h + hue_shift) % 1.0
                            hsv_img[i, j] = colorsys.hsv_to_rgb(h, s, v)
                        except:
                            hsv_img[i, j] = img_array[i, j]
            
            result_array = np.clip(hsv_img * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(result_array)
            
        except Exception as e:
            print(f"  ⚠️  Hue adjustment error: {e}")
            return image
    
    def apply_grayscale(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Apply grayscale conversion with validation."""
        try:
            if random.random() >= self.grayscale_prob:
                return image, True
                
            if not self._validate_image(image, "grayscale_input"):
                return image, False
            
            grayscale = image.convert('L')
            result = grayscale.convert('RGB')
            
            if not self._validate_image(result, "grayscale_output"):
                return image, False
                
            return result, True
            
        except Exception as e:
            print(f"  ⚠️  Grayscale error: {e}")
            return image, False
    
    def safe_save_image(self, image: Image.Image, save_path: str) -> bool:
        """
        Atomically save image to prevent corruption.
        
        Uses temporary file + move to ensure atomic operation.
        """
        try:
            if not self._validate_image(image, "save_validation"):
                self.stats['save_failures'] += 1
                return False
            
            # Create temporary file in same directory
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(
                suffix='.png', 
                dir=save_dir, 
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            
            # Save to temporary file
            image.save(temp_path, 'PNG', optimize=True)
            
            # Verify the saved file
            try:
                with Image.open(temp_path) as verify_img:
                    if verify_img.size != image.size:
                        os.unlink(temp_path)
                        self.stats['save_failures'] += 1
                        return False
            except Exception as e:
                print(f"  ⚠️  Save verification failed: {e}")
                os.unlink(temp_path)
                self.stats['save_failures'] += 1
                return False
            
            # Atomic move to final location
            shutil.move(temp_path, save_path)
            return True
            
        except Exception as e:
            print(f"  ⚠️  Safe save error: {e}")
            self.stats['save_failures'] += 1
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
            return False
    
    def augment_image_and_bboxes(
        self, 
        image: Image.Image, 
        bboxes: List[List[float]]
    ) -> Tuple[Image.Image, List[List[float]], Dict[str, Any]]:
        """
        Apply robust augmentation pipeline with comprehensive error handling.
        
        Returns:
            Tuple of (augmented_image, adjusted_bboxes, metadata_with_success_flag)
        """
        self.stats['total_attempts'] += 1
        
        try:
            if not self.should_augment():
                return image, bboxes, {"augmented": False, "success": True}
            
            if not self._validate_image(image, "augmentation_input"):
                self.stats['validation_failures'] += 1
                return image, bboxes, {"augmented": False, "success": False, "error": "Input validation failed"}
            
            augmented_image = image.copy()
            adjusted_bboxes = bboxes.copy()
            metadata = {"augmented": True, "transformations": [], "success": True}
            
            # Choose augmentation type
            augmentation_types = ["resize", "zoom", "color"]
            weights = [0.4, 0.3, 0.3]
            aug_type = random.choices(augmentation_types, weights=weights)[0]
            
            success = True
            
            # Apply geometric transformation
            if aug_type == "resize":
                scale_factor = random.uniform(*self.resize_range)
                augmented_image, adjusted_bboxes, success = self.resize_image_and_bboxes(
                    augmented_image, adjusted_bboxes, scale_factor
                )
                if success:
                    metadata["transformations"].append({
                        "type": "resize",
                        "scale_factor": scale_factor
                    })
            
            elif aug_type == "zoom":
                original_bbox_count = len(adjusted_bboxes)
                augmented_image, adjusted_bboxes, success = self.zoom_image_and_bboxes(
                    augmented_image, adjusted_bboxes
                )
                
                if success:
                    metadata["transformations"].append({
                        "type": "zoom",
                        "bbox_preserved": len(adjusted_bboxes) == original_bbox_count
                    })
                else:
                    metadata["transformations"].append({
                        "type": "zoom_failed_fallback_to_color"
                    })
            
            if not success:
                metadata["success"] = False
                return image, bboxes, metadata
            
            # Apply color augmentations (safer, always try)
            augmented_image, color_success = self.apply_color_augmentations(augmented_image)
            if color_success and random.random() < 0.3:
                metadata["transformations"].append({"type": "color_jitter"})
            
            # Apply grayscale
            augmented_image, gray_success = self.apply_grayscale(augmented_image)
            if gray_success and augmented_image.mode == 'RGB':
                # Check if actually grayscale
                img_array = np.array(augmented_image)
                if (len(img_array.shape) == 3 and 
                    np.allclose(img_array[:,:,0], img_array[:,:,1], atol=5) and 
                    np.allclose(img_array[:,:,1], img_array[:,:,2], atol=5)):
                    metadata["transformations"].append({"type": "grayscale"})
            
            # Final validation
            if not self._validate_image(augmented_image, "final_output"):
                self.stats['validation_failures'] += 1
                return image, bboxes, {"augmented": False, "success": False, "error": "Final validation failed"}
            
            self.stats['successful_augmentations'] += 1
            return augmented_image, adjusted_bboxes, metadata
            
        except Exception as e:
            print(f"  ⚠️  Augmentation pipeline error: {e}")
            metadata = {"augmented": False, "success": False, "error": str(e)}
            return image, bboxes, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics for debugging."""
        total = max(1, self.stats['total_attempts'])
        return {
            **self.stats,
            'success_rate': self.stats['successful_augmentations'] / total,
            'zoom_failure_rate': self.stats['zoom_failures'] / total,
            'resize_failure_rate': self.stats['resize_failures'] / total,
            'save_failure_rate': self.stats['save_failures'] / total,
            'validation_failure_rate': self.stats['validation_failures'] / total,
        }


def robust_validate_bboxes(
    bboxes: List[List[float]], 
    image_width: int, 
    image_height: int,
    min_area: int = 100
) -> List[List[float]]:
    """
    Robustly validate and filter bounding boxes with comprehensive checks.
    """
    valid_bboxes = []
    
    try:
        for bbox in bboxes:
            if not bbox or len(bbox) != 4:
                continue
                
            try:
                x_min, y_min, x_max, y_max = [float(coord) for coord in bbox]
                
                # Handle NaN/inf values
                if not all(math.isfinite(coord) for coord in [x_min, y_min, x_max, y_max]):
                    continue
                
                # Clamp to image boundaries
                x_min = max(0, min(x_min, image_width))
                y_min = max(0, min(y_min, image_height))
                x_max = max(0, min(x_max, image_width))
                y_max = max(0, min(y_max, image_height))
                
                # Check validity and minimum area
                if x_max > x_min and y_max > y_min:
                    area = (x_max - x_min) * (y_max - y_min)
                    if area >= min_area:
                        valid_bboxes.append([x_min, y_min, x_max, y_max])
                        
            except (ValueError, TypeError, OverflowError):
                continue
    
    except Exception as e:
        print(f"  ⚠️  BBox validation error: {e}")
    
    return valid_bboxes
