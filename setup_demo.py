#!/usr/bin/env python3
"""
Demo Setup Script for Image Annotation System
Creates sample data and directories for testing the annotation system.
"""

import os
import json
from pathlib import Path
import shutil

def setup_demo():
    """Set up demo environment for the annotation system"""
    
    print("🚀 Setting up Image Annotation System Demo...")
    
    # Create necessary directories
    directories = [
        "annotation_images",
        "annotation_results",
        "templates",
        "static/css",
        "static/js"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if we have existing images in the project
    existing_image_dirs = [
        "src/outputs",
        "src/notebooks/outputs", 
        "notebooks/outputs"
    ]
    
    copied_count = 0
    for img_dir in existing_image_dirs:
        if os.path.exists(img_dir):
            for filename in os.listdir(img_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    src_path = os.path.join(img_dir, filename)
                    dst_path = os.path.join("annotation_images", filename)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
    
    if copied_count > 0:
        print(f"✅ Copied {copied_count} existing images to annotation_images/")
    
    # Create initial progress file
    progress_data = {
        "classification_completed": [],
        "annotation_completed": [],
        "classification_in_progress": {},
        "annotation_in_progress": {},
        "images_with_artifacts": []
    }
    
    progress_file = Path("annotation_results/progress.json")
    if not progress_file.exists():
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print("✅ Created initial progress.json")
    
    # Create empty result files
    result_files = [
        "annotation_results/classification_results.json",
        "annotation_results/annotation_results.json"
    ]
    
    for result_file in result_files:
        result_path = Path(result_file)
        if not result_path.exists():
            with open(result_path, 'w') as f:
                json.dump([], f)
            print(f"✅ Created {result_file}")
    
    # Count images
    image_count = len([f for f in os.listdir("annotation_images") 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    
    print(f"\n📊 Setup Summary:")
    print(f"   • Images ready for annotation: {image_count}")
    print(f"   • Results directory: annotation_results/")
    print(f"   • Templates directory: templates/")
    
    if image_count == 0:
        print(f"\n⚠️  No images found! Please add images to annotation_images/")
        print(f"   Supported formats: PNG, JPG, JPEG, GIF, BMP")
        print(f"   You can also use the web upload interface at http://localhost:5000/upload")
    
    print(f"\n🚀 Ready to start!")
    print(f"   Run: python annotation_server.py")
    print(f"   Then visit: http://localhost:5000")

if __name__ == "__main__":
    setup_demo()
