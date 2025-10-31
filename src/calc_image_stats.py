#!/usr/bin/env python3
"""Calculate average image dimensions in a directory."""

import os
import sys
from PIL import Image

def calc_average_dimensions(directory, max_dimension=1024):
    """Calculate average width and height of images in directory and resize if needed."""
    import math
    
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.PNG', '.JPG', '.JPEG', '.BMP', '.GIF'}
    total_width = 0
    total_height = 0
    count = 0
    resized_count = 0
    
    # Maximum area based on max_dimension (assume square for max area)
    max_area = max_dimension * max_dimension
    
    # Track largest image
    max_area_found = 0
    max_width = 0
    max_height = 0
    largest_image_path = None
    
    for root, _, files in os.walk(directory):
        for f in files:
            if not any(f.endswith(ext) for ext in exts):
                continue
            
            filepath = os.path.join(root, f)
            im = Image.open(filepath)
            original_width, original_height = im.size
            width, height = original_width, original_height
            
            # Check if resizing is needed based on area
            current_area = width * height
            if current_area > max_area:
                # Calculate scale factor to maintain aspect ratio while limiting area
                # scale_factor = sqrt(max_area / current_area)
                scale_factor = math.sqrt(max_area / current_area)
                
                # Calculate new dimensions maintaining aspect ratio
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize the image
                im = im.resize((new_width, new_height), Image.Resampling.LANCZOS)
                width, height = new_width, new_height
                
                # Convert RGBA to RGB for JPEG files
                file_lower = filepath.lower()
                if file_lower.endswith(('.jpg', '.jpeg')):
                    if im.mode == 'RGBA':
                        rgb_im = Image.new('RGB', im.size, (255, 255, 255))
                        rgb_im.paste(im, mask=im.split()[3])
                        im = rgb_im
                    elif im.mode not in ('RGB', 'L'):
                        im = im.convert('RGB')
                
                # Save the resized image (handle different formats)
                save_kwargs = {}
                if file_lower.endswith(('.jpg', '.jpeg')):
                    save_kwargs['quality'] = 95
                    save_kwargs['optimize'] = True
                elif file_lower.endswith('.png'):
                    save_kwargs['optimize'] = True
                
                im.save(filepath, **save_kwargs)
                resized_count += 1
                print(f"Resized {filepath}: {original_width}x{original_height} -> {width}x{height}")
            
            final_area = width * height
            
            total_width += width
            total_height += height
            count += 1
            
            # Track largest image by area
            if final_area > max_area_found:
                max_area_found = final_area
                max_width = width
                max_height = height
                largest_image_path = filepath
            
            im.close()
    
    if count == 0:
        print(f"No images found in {directory}")
        return
    
    avg_width = total_width // count
    avg_height = total_height // count
    
    print(f"Average width: {avg_width}")
    print(f"Average height: {avg_height}")
    print(f"Image count: {count}")
    print(f"Resized images: {resized_count}")
    print(f"Largest image: {max_width}x{max_height} ({largest_image_path})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_image_stats.py <directory>")
        sys.exit(1)
    
    calc_average_dimensions(sys.argv[1])

