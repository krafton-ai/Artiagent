#!/usr/bin/env python3
"""
Image Annotation Server
A Flask-based web server for collaborative image annotation with artifact detection.
"""

import os
import json
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from werkzeug.utils import secure_filename
import fcntl

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Configuration
IMAGES_DIR = Path("annotation_images")
RESULTS_DIR = Path("annotation_results")
CLASSIFICATION_RESULTS = RESULTS_DIR / "classification_results.json"
ANNOTATION_RESULTS = RESULTS_DIR / "annotation_results.json"
WORK_ASSIGNMENT_FILE = RESULTS_DIR / "work_assignments.json"
PROGRESS_FILE = RESULTS_DIR / "progress.json"

# Ensure directories exist
IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Thread lock for file operations
file_lock = threading.Lock()

class WorkCoordinator:
    """Handles work assignment and progress tracking"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress data"""
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE, 'r') as f:
                    self.progress = json.load(f)
                
                # Migrate old progress files to new format
                if "images_unsure" not in self.progress:
                    self.progress["images_unsure"] = []
                if "admin_classification_completed" not in self.progress:
                    self.progress["admin_classification_completed"] = []
                if "admin_annotation_completed" not in self.progress:
                    self.progress["admin_annotation_completed"] = []
                    self.save_progress()  # Save migrated format
            else:
                self.progress = {
                    "classification_completed": [],
                    "annotation_completed": [],
                    "classification_in_progress": {},
                    "annotation_in_progress": {},
                    "images_with_artifacts": [],
                    "images_unsure": [],
                    "admin_classification_completed": [],
                    "admin_annotation_completed": []
                }
                self.save_progress()
        except Exception as e:
            print(f"Error loading progress: {e}")
            self.progress = {
                "classification_completed": [],
                "annotation_completed": [],
                "classification_in_progress": {},
                "annotation_in_progress": {},
                "images_with_artifacts": [],
                "images_unsure": [],
                "admin_classification_completed": [],
                "admin_annotation_completed": []
            }
    
    def save_progress(self):
        """Save progress data to file"""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_next_classification_image(self, user_id):
        """Get next image for classification"""
        with self.lock:
            # Get all images
            all_images = [f for f in os.listdir(IMAGES_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            # Remove completed and in-progress images
            available_images = [img for img in all_images 
                              if img not in self.progress["classification_completed"] 
                              and img not in self.progress["classification_in_progress"].values()]
            
            if not available_images:
                return None
            
            # Assign next image
            next_image = available_images[0]
            self.progress["classification_in_progress"][user_id] = next_image
            self.save_progress()
            return next_image
    
    def complete_classification(self, user_id, image_name, classification_result):
        """Mark classification as complete"""
        with self.lock:
            # Remove from in-progress
            if user_id in self.progress["classification_in_progress"]:
                del self.progress["classification_in_progress"][user_id]
            
            # Add to completed
            if image_name not in self.progress["classification_completed"]:
                self.progress["classification_completed"].append(image_name)
            
            # Track images by classification result
            if classification_result == True and image_name not in self.progress["images_with_artifacts"]:
                self.progress["images_with_artifacts"].append(image_name)
            elif classification_result == 'unsure' and image_name not in self.progress["images_unsure"]:
                self.progress["images_unsure"].append(image_name)
            
            self.save_progress()
    
    def get_next_annotation_image(self, user_id):
        """Get next image for annotation (only artifact images)"""
        with self.lock:
            # Get artifact images that need annotation
            available_images = [img for img in self.progress["images_with_artifacts"]
                              if img not in self.progress["annotation_completed"]
                              and img not in self.progress["annotation_in_progress"].values()]
            
            if not available_images:
                return None
            
            # Assign next image
            next_image = available_images[0]
            self.progress["annotation_in_progress"][user_id] = next_image
            self.save_progress()
            return next_image
    
    def complete_annotation(self, user_id, image_name):
        """Mark annotation as complete"""
        with self.lock:
            # Remove from in-progress
            if user_id in self.progress["annotation_in_progress"]:
                del self.progress["annotation_in_progress"][user_id]
            
            # Add to completed
            if image_name not in self.progress["annotation_completed"]:
                self.progress["annotation_completed"].append(image_name)
            
            self.save_progress()
    
    def get_current_image(self, user_id, task_type):
        """Get currently assigned image for user"""
        if task_type == "classification":
            return self.progress["classification_in_progress"].get(user_id)
        else:
            return self.progress["annotation_in_progress"].get(user_id)
    
    def get_statistics(self):
        """Get current progress statistics"""
        all_images = [f for f in os.listdir(IMAGES_DIR) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        # Calculate admin annotation available count (admin classified as artifacts but not yet annotated)
        admin_annotation_available = 0
        if CLASSIFICATION_RESULTS.exists():
            try:
                with open(CLASSIFICATION_RESULTS, 'r') as f:
                    classifications = json.load(f)
                    # Find admin-classified artifact images
                    admin_artifacts = set()
                    for c in classifications:
                        if (c.get('task_type') == 'admin_classification' and 
                            c.get('has_artifact') == True):
                            admin_artifacts.add(c.get('image_name'))
                    
                    # Subtract already annotated by admin
                    admin_completed = set(self.progress.get("admin_annotation_completed", []))
                    admin_annotation_available = len(admin_artifacts - admin_completed)
            except Exception as e:
                print(f"Error calculating admin annotation statistics: {e}")
        
        return {
            "total_images": len(all_images),
            "classification_completed": len(self.progress["classification_completed"]),
            "classification_remaining": len(all_images) - len(self.progress["classification_completed"]),
            "images_with_artifacts": len(self.progress["images_with_artifacts"]),
            "images_unsure": len(self.progress.get("images_unsure", [])),
            "images_no_artifacts": len(self.progress["classification_completed"]) - len(self.progress["images_with_artifacts"]) - len(self.progress.get("images_unsure", [])),
            "annotation_completed": len(self.progress["annotation_completed"]),
            "annotation_remaining": len(self.progress["images_with_artifacts"]) - len(self.progress["annotation_completed"]),
            "classification_in_progress": len(self.progress["classification_in_progress"]),
            "annotation_in_progress": len(self.progress["annotation_in_progress"]),
            "admin_classification_completed": len(self.progress.get("admin_classification_completed", [])),
            "admin_annotation_completed": len(self.progress.get("admin_annotation_completed", [])),
            "admin_annotation_available": admin_annotation_available
        }
    
    def get_next_admin_classification_image(self, user_id):
        """Get next unsure image for admin classification"""
        with self.lock:
            # Get unsure images that haven't been admin-classified yet
            available_images = [img for img in self.progress.get("images_unsure", [])
                              if img not in self.progress.get("admin_classification_completed", [])]
            
            if not available_images:
                return None
                
            return available_images[0]
    
    def complete_admin_classification(self, user_id, image_name, has_artifact):
        """Mark admin classification as complete"""
        with self.lock:
            # Add to admin completed
            if image_name not in self.progress.get("admin_classification_completed", []):
                if "admin_classification_completed" not in self.progress:
                    self.progress["admin_classification_completed"] = []
                self.progress["admin_classification_completed"].append(image_name)
            
            # If admin marks it as having artifacts, add to annotation queue
            if has_artifact and image_name not in self.progress["images_with_artifacts"]:
                self.progress["images_with_artifacts"].append(image_name)
            
            self.save_progress()
    
    def get_next_admin_annotation_image(self, user_id):
        """Get next image for admin annotation (admin-classified artifact images)"""
        with self.lock:
            # Get admin-classified artifact images that need annotation
            admin_classified_artifacts = []
            
            # Find images that admin classified as having artifacts
            for img in self.progress.get("admin_classification_completed", []):
                # Check if this image was classified as having artifacts by admin
                # We need to check the actual classification results
                try:
                    if CLASSIFICATION_RESULTS.exists():
                        with open(CLASSIFICATION_RESULTS, 'r') as f:
                            classifications = json.load(f)
                            # Find the most recent admin classification for this image
                            admin_classifications = [
                                c for c in classifications 
                                if (c.get('image_name') == img and 
                                    c.get('task_type') == 'admin_classification' and
                                    c.get('has_artifact') == True)
                            ]
                            if admin_classifications:
                                admin_classified_artifacts.append(img)
                except Exception as e:
                    print(f"Error checking admin classifications: {e}")
            
            # Filter out already admin-annotated images
            available_images = [
                img for img in admin_classified_artifacts
                if img not in self.progress.get("admin_annotation_completed", [])
            ]
            
            if not available_images:
                return None
            
            return available_images[0]
    
    def complete_admin_annotation(self, user_id, image_name):
        """Mark admin annotation as complete"""
        with self.lock:
            # Add to admin annotation completed
            if image_name not in self.progress.get("admin_annotation_completed", []):
                if "admin_annotation_completed" not in self.progress:
                    self.progress["admin_annotation_completed"] = []
                self.progress["admin_annotation_completed"].append(image_name)
            
            self.save_progress()

# Initialize work coordinator
coordinator = WorkCoordinator()

def save_result_safely(filename, data):
    """Save data to JSON file safely with file locking"""
    with file_lock:
        try:
            # Read existing data
            existing_data = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new data
            existing_data.append(data)
            
            # Write back
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving result: {e}")
            return False

@app.route('/')
def home():
    """Home page - choose task type"""
    return render_template('home.html')

@app.route('/classify')
def classify():
    """Classification interface"""
    # Generate or get user ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    return render_template('classify.html')

@app.route('/annotate')
def annotate():
    """Annotation interface"""
    # Generate or get user ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    return render_template('annotate.html')

@app.route('/admin')
def admin():
    """Admin dashboard"""
    stats = coordinator.get_statistics()
    return render_template('admin.html', stats=stats)

@app.route('/admin/classify')
def admin_classify():
    """Admin classification interface for unsure images"""
    return render_template('admin_classify.html')

@app.route('/admin/annotate')
def admin_annotate():
    """Admin annotation interface for admin-classified images"""
    if 'user_id' not in session:
        session['user_id'] = 'admin'
    
    return render_template('admin_annotate.html')

@app.route('/api/get-admin-image/<task_type>')
def get_admin_image(task_type):
    """Get next image for admin classification or annotation"""
    user_id = session.get('user_id', 'admin')
    
    if task_type == "classification":
        image_name = coordinator.get_next_admin_classification_image(user_id)
    elif task_type == "annotation":
        image_name = coordinator.get_next_admin_annotation_image(user_id)
    else:
        return jsonify({"error": f"Unknown task type: {task_type}"}), 400
    
    if not image_name:
        return jsonify({"completed": True, "message": f"All admin {task_type} tasks completed!"})
    
    return jsonify({
        "completed": False,
        "image_name": image_name,
        "image_url": f"/images/{image_name}"
    })

@app.route('/api/submit-admin-classification', methods=['POST'])
def submit_admin_classification():
    """Submit admin classification result"""
    user_id = session.get('user_id', 'admin')
    
    data = request.json
    image_name = data.get('image_name')
    has_artifact = data.get('has_artifact')
    
    if image_name is None or has_artifact is None:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Save admin classification result
    result_data = {
        "image_name": image_name,
        "has_artifact": has_artifact,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "task_type": "admin_classification"
    }
    
    if save_result_safely(CLASSIFICATION_RESULTS, result_data):
        coordinator.complete_admin_classification(user_id, image_name, has_artifact)
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to save result"}), 500

@app.route('/api/submit-admin-annotation', methods=['POST'])
def submit_admin_annotation():
    """Submit admin annotation result"""
    user_id = session.get('user_id', 'admin')
    
    data = request.json
    image_name = data.get('image_name')
    bboxes = data.get('bboxes', [])
    
    if image_name is None:
        return jsonify({"error": "Missing image_name"}), 400
    
    if not bboxes:
        return jsonify({"error": "At least one bounding box is required"}), 400
    
    # Save admin annotation result
    result_data = {
        "image_name": image_name,
        "global_explanation": "",  # Admin annotations don't require global explanation
        "bboxes": bboxes,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "task_type": "admin_annotation"
    }
    
    if save_result_safely(ANNOTATION_RESULTS, result_data):
        coordinator.complete_admin_annotation(user_id, image_name)
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to save result"}), 500

@app.route('/api/get-image/<task_type>')
def get_image(task_type):
    """Get next image for classification or annotation"""
    user_id = session.get('user_id', str(uuid.uuid4()))
    session['user_id'] = user_id
    
    if task_type == "classification":
        image_name = coordinator.get_next_classification_image(user_id)
    elif task_type == "annotation":
        image_name = coordinator.get_next_annotation_image(user_id)
    else:
        return jsonify({"error": "Invalid task type"}), 400
    
    if not image_name:
        return jsonify({"completed": True, "message": f"All {task_type} tasks completed!"})
    
    return jsonify({
        "completed": False,
        "image_name": image_name,
        "image_url": f"/images/{image_name}"
    })

@app.route('/api/submit-classification', methods=['POST'])
def submit_classification():
    """Submit classification result"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "No user session"}), 400
    
    data = request.json
    image_name = data.get('image_name')
    classification_result = data.get('has_artifact')  # Can be True, False, or 'unsure'
    
    if image_name is None or classification_result is None:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Save classification result
    result_data = {
        "image_name": image_name,
        "has_artifact": classification_result,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "task_type": "classification"
    }
    
    if save_result_safely(CLASSIFICATION_RESULTS, result_data):
        coordinator.complete_classification(user_id, image_name, classification_result)
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to save result"}), 500

@app.route('/api/submit-annotation', methods=['POST'])
def submit_annotation():
    """Submit annotation result"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "No user session"}), 400
    
    data = request.json
    image_name = data.get('image_name')
    global_explanation = data.get('global_explanation', '')
    bboxes = data.get('bboxes', [])
    
    if not image_name:
        return jsonify({"error": "Missing image_name"}), 400
    
    # Save annotation result
    result_data = {
        "image_name": image_name,
        "global_explanation": global_explanation,
        "bboxes": bboxes,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "task_type": "annotation"
    }
    
    if save_result_safely(ANNOTATION_RESULTS, result_data):
        coordinator.complete_annotation(user_id, image_name)
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to save result"}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get current statistics"""
    return jsonify(coordinator.get_statistics())

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images"""
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/examples/images/<filename>')
def serve_example_image(filename):
    """Serve example images"""
    return send_from_directory(Path("examples/images"), filename)

@app.route('/api/examples')
def get_examples():
    """Get all example annotations"""
    examples_dir = Path("examples/annotations")
    examples = []
    
    if examples_dir.exists():
        for json_file in examples_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    example_data = json.load(f)
                    examples.append(example_data)
            except Exception as e:
                print(f"Error loading example {json_file}: {e}")
    
    return jsonify(examples)

@app.route('/api/prompts')
def get_prompts():
    """Get prompt data for images"""
    try:
        # Look for JSON files in the images directory
        prompts_file = None
        for json_file in IMAGES_DIR.glob("*.json"):
            prompts_file = json_file
            break
        
        if not prompts_file or not prompts_file.exists():
            print("No JSON file found in images directory")
            return jsonify({"prompts": {}, "debug": "No JSON file found"})
        
        print(f"Found prompts file: {prompts_file}")
        
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        # Create a mapping from image names to prompts
        prompts_map = {}
        
        if 'images' in data:
            print(f"Processing {len(data['images'])} image entries")
            
            for i, image_data in enumerate(data['images']):
                # Try different ways to match images to prompts
                # Method 1: Use caption_index to match with image filenames
                caption_index = image_data.get('caption_index', i)
                prompt = image_data.get('prompt', '')
                
                # Look for images that might match this index
                possible_names = [
                    f"image_{caption_index:04d}_flux-schnell.png",
                    f"image_{caption_index:04d}_flux-schnell.jpg",
                    f"image_{caption_index:04d}.png", 
                    f"image_{caption_index:04d}.jpg",
                    f"image_{caption_index:03d}_flux-schnell.png",
                    f"image_{caption_index}.png",
                    f"image_{caption_index}.jpg",
                    f"{caption_index:04d}.png",
                    f"{caption_index:04d}.jpg"
                ]
                
                found_match = False
                for name in possible_names:
                    if (IMAGES_DIR / name).exists():
                        prompts_map[name] = prompt
                        print(f"Matched caption_index {caption_index} → {name}: {prompt[:50]}...")
                        found_match = True
                        break
                
                if not found_match:
                    print(f"No match found for caption_index {caption_index}, tried: {possible_names[:3]}...")
        
        print(f"Created prompts map with {len(prompts_map)} entries")
        return jsonify({"prompts": prompts_map, "debug": f"Loaded {len(prompts_map)} prompts"})
        
    except Exception as e:
        print(f"Error loading prompts: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"prompts": {}, "error": str(e)})

@app.route('/upload', methods=['GET', 'POST'])
def upload_images():
    """Upload images (for admin use)"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    uploaded_files = request.files.getlist("images")
    success_count = 0
    
    for file in uploaded_files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = IMAGES_DIR / filename
                file.save(file_path)
                success_count += 1
    
    return jsonify({
        "success": True,
        "message": f"Uploaded {success_count} images successfully"
    })

@app.route('/api/download/classification')
def download_classification_results():
    """Download classification results"""
    try:
        return send_from_directory(RESULTS_DIR, 'classification_results.json', as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Classification results not found"}), 404

@app.route('/api/download/annotation')
def download_annotation_results():
    """Download annotation results"""
    try:
        return send_from_directory(RESULTS_DIR, 'annotation_results.json', as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Annotation results not found"}), 404

if __name__ == '__main__':
    print("Starting Image Annotation Server...")
    print(f"Images directory: {IMAGES_DIR.absolute()}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print("\nAccess the application at:")
    print("- Classification: http://localhost:5000/classify")
    print("- Annotation: http://localhost:5000/annotate")
    print("- Admin Dashboard: http://localhost:5000/admin")
    print("- Upload Images: http://localhost:5000/upload")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
