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
import logging
from functools import wraps
from collections import defaultdict
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Global caches for stability
_prompt_cache = {}
_prompt_cache_time = 0
_examples_cache = {}
_examples_cache_time = 0
CACHE_DURATION = 300  # 5 minutes

# Request throttling
request_counts = defaultdict(list)
REQUEST_LIMIT = 50  # requests per minute per user
REQUEST_WINDOW = 60

# Configuration
IMAGES_DIR = Path("annotation_images")
RESULTS_DIR = Path("annotation_results")
CLASSIFICATION_RESULTS = RESULTS_DIR / "classification_results.json"
ANNOTATION_RESULTS = RESULTS_DIR / "annotation_results.json"
WORK_ASSIGNMENT_FILE = RESULTS_DIR / "work_assignments.json"
PROGRESS_FILE = RESULTS_DIR / "progress.json"

# Assignment timeout configuration (in seconds)
ASSIGNMENT_TIMEOUT = 10 * 60  # 10 minutes - configurable timeout for stale assignments

# Ensure directories exist
IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Thread lock for file operations
file_lock = threading.Lock()

def throttle_requests(func):
    """Decorator to throttle requests per user"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        user_id = session.get('user_id', request.remote_addr)
        current_time = time.time()
        
        # Clean old requests
        request_counts[user_id] = [
            req_time for req_time in request_counts[user_id]
            if current_time - req_time < REQUEST_WINDOW
        ]
        
        # Check rate limit
        if len(request_counts[user_id]) >= REQUEST_LIMIT:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return jsonify({"error": "Rate limit exceeded. Please slow down."}), 429
        
        # Add current request
        request_counts[user_id].append(current_time)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    return wrapper

def with_retry(max_retries=3, delay=0.1):
    """Decorator to retry operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def ensure_session(func):
    """Decorator to ensure user has a valid session"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
            session['created_at'] = time.time()
        
        # Validate session age (expire after 24 hours)
        if time.time() - session.get('created_at', 0) > 86400:
            session.clear()
            session['user_id'] = str(uuid.uuid4())
            session['created_at'] = time.time()
        
        return func(*args, **kwargs)
    return wrapper

class WorkCoordinator:
    """Handles work assignment and progress tracking with automatic timeout management"""
    
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
                
                # Add individual annotator progress tracking
                if "user_completions" not in self.progress:
                    self.progress["user_completions"] = {
                        "classification": {},  # user_id -> {"images": [], "ip": ""}
                        "annotation": {}        # user_id -> {"images": [], "ip": ""}
                    }
                
                # Add IP to user mapping for session consolidation
                if "ip_to_users" not in self.progress:
                    self.progress["ip_to_users"] = {}  # ip -> [user_ids]
                
                # Migrate assignment format to include timestamps
                self._migrate_assignment_format()
                
                self.save_progress()  # Save migrated format
            else:
                self.progress = {
                    "classification_completed": [],
                    "annotation_completed": [],
                    "classification_in_progress": {},  # user_id -> {"image": image_name, "timestamp": timestamp}
                    "annotation_in_progress": {},      # user_id -> {"image": image_name, "timestamp": timestamp}
                    "images_with_artifacts": [],
                    "images_unsure": [],
                    "admin_classification_completed": [],
                    "admin_annotation_completed": [],
                    "user_completions": {
                        "classification": {},
                        "annotation": {}
                    },
                    "ip_to_users": {}
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
                "admin_annotation_completed": [],
                "user_completions": {
                    "classification": {},
                    "annotation": {}
                },
                "ip_to_users": {}
            }
    
    def save_progress(self):
        """Save progress data to file"""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _migrate_assignment_format(self):
        """Migrate old assignment format (user_id -> image) to new format (user_id -> {image, timestamp})"""
        current_time = time.time()
        
        for task_type in ["classification_in_progress", "annotation_in_progress"]:
            if task_type in self.progress:
                assignments = self.progress[task_type]
                migrated = {}
                
                for user_id, assignment in assignments.items():
                    # Check if already in new format
                    if isinstance(assignment, dict) and "image" in assignment and "timestamp" in assignment:
                        migrated[user_id] = assignment
                    else:
                        # Old format: user_id -> image_name
                        # Migrate to new format with current timestamp
                        migrated[user_id] = {
                            "image": assignment,
                            "timestamp": current_time
                        }
                        logger.info(f"Migrated {task_type} assignment for user {user_id}: {assignment}")
                
                self.progress[task_type] = migrated
    
    def _cleanup_stale_assignments(self):
        """Remove assignments that are older than ASSIGNMENT_TIMEOUT seconds"""
        current_time = time.time()
        cleaned_count = 0
        
        for task_type in ["classification_in_progress", "annotation_in_progress"]:
            if task_type not in self.progress:
                continue
                
            assignments = self.progress[task_type]
            stale_users = []
            
            for user_id, assignment in assignments.items():
                # Handle both old and new format during migration period
                if isinstance(assignment, dict):
                    timestamp = assignment.get("timestamp", 0)
                    image_name = assignment.get("image", "unknown")
                else:
                    # Old format fallback - assume it's stale since we can't track time
                    timestamp = 0
                    image_name = assignment
                
                # Check if assignment is stale
                if current_time - timestamp > ASSIGNMENT_TIMEOUT:
                    stale_users.append(user_id)
                    logger.info(f"Cleaning up stale {task_type} assignment: user {user_id}, image {image_name}, age {int(current_time - timestamp)}s")
                    cleaned_count += 1
            
            # Remove stale assignments
            for user_id in stale_users:
                del assignments[user_id]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale assignments")
            self.save_progress()
        
        return cleaned_count
    
    def cleanup_stale_assignments_manual(self):
        """Manual cleanup trigger for testing and admin use"""
        with self.lock:
            return self._cleanup_stale_assignments()
    
    def get_next_classification_image(self, user_id):
        """Get next image for classification with automatic stale assignment cleanup"""
        with self.lock:
            # Clean up stale assignments first
            self._cleanup_stale_assignments()
            
            # Get all images
            all_images = [f for f in os.listdir(IMAGES_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            # Get currently assigned images (extract from new format)
            assigned_images = set()
            for assignment in self.progress["classification_in_progress"].values():
                if isinstance(assignment, dict):
                    assigned_images.add(assignment.get("image"))
                else:
                    # Handle old format during migration
                    assigned_images.add(assignment)
            
            # Remove completed and in-progress images
            available_images = [img for img in all_images 
                              if img not in self.progress["classification_completed"] 
                              and img not in assigned_images]
            
            if not available_images:
                return None
            
            # Assign next image with timestamp
            next_image = available_images[0]
            self.progress["classification_in_progress"][user_id] = {
                "image": next_image,
                "timestamp": time.time()
            }
            self.save_progress()
            logger.info(f"Assigned classification image {next_image} to user {user_id}")
            return next_image
    
    def complete_classification(self, user_id, image_name, classification_result, user_ip=None):
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
            
            # Track individual user completion
            self._track_user_completion(user_id, image_name, "classification", user_ip)
            
            self.save_progress()
    
    def get_next_annotation_image(self, user_id):
        """Get next image for annotation (only artifact images) with automatic stale assignment cleanup"""
        with self.lock:
            # Clean up stale assignments first
            self._cleanup_stale_assignments()
            
            # Get currently assigned images (extract from new format)
            assigned_images = set()
            for assignment in self.progress["annotation_in_progress"].values():
                if isinstance(assignment, dict):
                    assigned_images.add(assignment.get("image"))
                else:
                    # Handle old format during migration
                    assigned_images.add(assignment)
            
            # Get artifact images that need annotation
            available_images = [img for img in self.progress["images_with_artifacts"]
                              if img not in self.progress["annotation_completed"]
                              and img not in assigned_images]
            
            if not available_images:
                return None
            
            # Assign next image with timestamp
            next_image = available_images[0]
            self.progress["annotation_in_progress"][user_id] = {
                "image": next_image,
                "timestamp": time.time()
            }
            self.save_progress()
            logger.info(f"Assigned annotation image {next_image} to user {user_id}")
            return next_image
    
    def complete_annotation(self, user_id, image_name, user_ip=None):
        """Mark annotation as complete"""
        with self.lock:
            # Remove from in-progress
            if user_id in self.progress["annotation_in_progress"]:
                del self.progress["annotation_in_progress"][user_id]
            
            # Add to completed
            if image_name not in self.progress["annotation_completed"]:
                self.progress["annotation_completed"].append(image_name)
            
            # Track individual user completion
            self._track_user_completion(user_id, image_name, "annotation", user_ip)
            
            self.save_progress()
    
    def get_current_image(self, user_id, task_type):
        """Get currently assigned image for user"""
        if task_type == "classification":
            assignment = self.progress["classification_in_progress"].get(user_id)
        else:
            assignment = self.progress["annotation_in_progress"].get(user_id)
        
        # Handle both old and new assignment formats
        if isinstance(assignment, dict):
            return assignment.get("image")
        else:
            return assignment
    
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
    
    def _track_user_completion(self, user_id, image_name, task_type, user_ip=None):
        """Track individual user completion with IP association"""
        # Initialize user completion data if not exists
        if user_id not in self.progress["user_completions"][task_type]:
            self.progress["user_completions"][task_type][user_id] = {
                "images": [],
                "ip": user_ip or "unknown"
            }
        
        # Add image to user's completed list if not already there
        user_data = self.progress["user_completions"][task_type][user_id]
        if image_name not in user_data["images"]:
            user_data["images"].append(image_name)
        
        # Update IP if provided and different
        if user_ip and user_data["ip"] != user_ip:
            user_data["ip"] = user_ip
        
        # Track IP to users mapping
        if user_ip:
            if user_ip not in self.progress["ip_to_users"]:
                self.progress["ip_to_users"][user_ip] = []
            if user_id not in self.progress["ip_to_users"][user_ip]:
                self.progress["ip_to_users"][user_ip].append(user_id)
    
    def get_individual_annotator_stats(self):
        """Get individual annotator progress statistics grouped by IP"""
        stats = {
            "by_ip": {},
            "by_user": {
                "classification": {},
                "annotation": {}
            }
        }
        
        # Group by IP address
        for ip, user_ids in self.progress.get("ip_to_users", {}).items():
            ip_stats = {
                "classification_count": 0,
                "annotation_count": 0,
                "users": [],
                "last_active_user": None
            }
            
            # Aggregate stats for all users from this IP
            classification_images = set()
            annotation_images = set()
            
            for user_id in user_ids:
                # Classification stats
                user_classification = self.progress["user_completions"]["classification"].get(user_id, {})
                if user_classification:
                    classification_images.update(user_classification.get("images", []))
                    ip_stats["users"].append(user_id)
                    ip_stats["last_active_user"] = user_id  # Last user will be the most recent
                
                # Annotation stats  
                user_annotation = self.progress["user_completions"]["annotation"].get(user_id, {})
                if user_annotation:
                    annotation_images.update(user_annotation.get("images", []))
                    if user_id not in ip_stats["users"]:
                        ip_stats["users"].append(user_id)
                    ip_stats["last_active_user"] = user_id
            
            ip_stats["classification_count"] = len(classification_images)
            ip_stats["annotation_count"] = len(annotation_images)
            stats["by_ip"][ip] = ip_stats
        
        # Individual user stats
        for task_type in ["classification", "annotation"]:
            for user_id, user_data in self.progress["user_completions"][task_type].items():
                stats["by_user"][task_type][user_id] = {
                    "count": len(user_data.get("images", [])),
                    "ip": user_data.get("ip", "unknown"),
                    "images": user_data.get("images", [])
                }
        
        return stats

# Initialize work coordinator
coordinator = WorkCoordinator()

@with_retry(max_retries=5, delay=0.1)
def save_result_safely(filename, data):
    """Save data to JSON file safely with file locking and retry mechanism"""
    with file_lock:
        temp_filename = f"{filename}.tmp"
        backup_filename = f"{filename}.backup"
        
        try:
            # Create backup of existing file
            if os.path.exists(filename):
                import shutil
                shutil.copy2(filename, backup_filename)
            
            # Read existing data
            existing_data = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Validate data structure
            if not isinstance(existing_data, list):
                existing_data = []
            
            # Add new data
            existing_data.append(data)
            
            # Write to temporary file first
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            import shutil
            shutil.move(temp_filename, filename)
            
            # Remove backup on success
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
            
            logger.info(f"Successfully saved result to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving result to {filename}: {e}")
            
            # Cleanup temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
            
            # Restore from backup if needed
            if os.path.exists(backup_filename) and not os.path.exists(filename):
                try:
                    import shutil
                    shutil.move(backup_filename, filename)
                    logger.info(f"Restored {filename} from backup")
                except:
                    pass
            
            raise  # Let retry mechanism handle this

@app.route('/')
def home():
    """Home page - choose task type"""
    return render_template('home.html')

@app.route('/guidelines')
def guidelines():
    """Redirect to English guidelines by default"""
    return render_template('guidelines_eng.html')

@app.route('/guidelines_eng')
def guidelines_eng():
    """English guidelines page"""
    return render_template('guidelines_eng.html')

@app.route('/guidelines_kor')
def guidelines_kor():
    """Korean guidelines page"""
    return render_template('guidelines_kor.html')

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
@ensure_session
@throttle_requests
def get_image(task_type):
    """Get next image for classification or annotation"""
    user_id = session['user_id']
    
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
@ensure_session
@throttle_requests
def submit_classification():
    """Submit classification result"""
    user_id = session['user_id']
    
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
        coordinator.complete_classification(user_id, image_name, classification_result, request.remote_addr)
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to save result"}), 500

@app.route('/api/submit-annotation', methods=['POST'])
@ensure_session
@throttle_requests
def submit_annotation():
    """Submit annotation result"""
    user_id = session['user_id']
    
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
        coordinator.complete_annotation(user_id, image_name, request.remote_addr)
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to save result"}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get current statistics"""
    return jsonify(coordinator.get_statistics())

@app.route('/api/individual-progress')
def get_individual_progress():
    """Get individual annotator progress statistics"""
    return jsonify(coordinator.get_individual_annotator_stats())

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images"""
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/examples/images/<filename>')
def serve_example_image(filename):
    """Serve example images"""
    return send_from_directory(Path("examples/images"), filename)

@with_retry(max_retries=3, delay=0.1)
def load_examples_from_disk():
    """Load examples from disk with caching"""
    global _examples_cache, _examples_cache_time
    
    current_time = time.time()
    
    # Return cached data if still valid
    if _examples_cache and (current_time - _examples_cache_time) < CACHE_DURATION:
        logger.debug("Returning cached examples")
        return _examples_cache
    
    logger.info("Loading examples from disk")
    examples_dir = Path("examples/annotations")
    examples = []
    
    if examples_dir.exists():
        for json_file in sorted(examples_dir.glob("*.json")):  # Sort for consistency
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    example_data = json.load(f)
                    # Validate example structure
                    if isinstance(example_data, dict) and 'image_name' in example_data:
                        examples.append(example_data)
                        logger.debug(f"Loaded example: {example_data.get('image_name')}")
            except Exception as e:
                logger.error(f"Error loading example {json_file}: {e}")
    
    # Update cache
    _examples_cache = examples
    _examples_cache_time = current_time
    
    logger.info(f"Successfully loaded {len(examples)} examples")
    return examples

@app.route('/api/examples')
@ensure_session
@throttle_requests
def get_examples():
    """Get all example annotations with caching"""
    try:
        return jsonify(load_examples_from_disk())
    except Exception as e:
        logger.error(f"Error in get_examples: {e}")
        return jsonify([]), 500

@app.route('/api/classification-examples')
def get_classification_examples():
    """Get examples for classification - showing has artifacts vs no artifacts"""
    # TODO: Replace with actual examples - these are temporary samples
    examples = {
        "has_artifacts": [
            {
                "image_name": "artifact_example1.png",
                "description": "Vector illustration of a cartoon businessman and woman, both wearing black suits and glasses, giving the thumbs up. A long arm is visible in the foreground.",
                "artifact_type": "Addition",
                "explanation": "extra finger on the foreground arm"
            },
            {
                "image_name": "artifact_example2.png", 
                "description": "A person is standing outside with an open umbrella covering them.",
                "artifact_type": "Distortion",
                "explanation": "physically distorted umbrella, with the parts connected in an unnatural way"
            },
            {
                "image_name": "artifact_example3.png",
                "description": "A man standing at a table with two pizzas.",
                "artifact_type": "Removal",
                "explanation": "fingers missing from the hand"
            },
            {
                "image_name": "artifact_example4.png",
                "description": "A marketplace with several people wandering around and riding bikes.",
                "artifact_type": "Fusion, Distortion",
                "explanation": "background features indistinguishable and distorted, person in the middle has the hat merged with the body"
            },
            {
                "image_name": "artifact_example5.png",
                "description": "Three lambs standing on a dirt road.",
                "artifact_type": "Fusion",
                "explanation": "two lambs' bodies are merged into one head"
            },
            {
                "image_name": "artifact_example6.png",
                "description": "A blurry photo of a skateboarder doing tricks while people watch.",
                "artifact_type": "Addition",
                "explanation": "extra leg to the girl among the crowd"
            }
        ],
        "no_artifacts": [
            {
                "image_name": "clean_example1.png",
                "type": "Physical abnoramlity",
                "description": "There's a plane parked in a court yard.",
                "explanation": "unrealistic situations. Not classified as artifacts in this task."
            },
            {
                "image_name": "clean_example2.png",
                "type": "Stylistic error",
                "description": "Video game icon design depicting a tribal mask.",
                "explanation": "Stylistic errors are not classified as artifacts in this task. Cartoonish or painting style images are only classified as artifacts if there are structural errors."
            },
            {
                "image_name": "clean_example3.png",
                "type": "Text distortion",
                "description": "A glass bottle with a note inside of it.",
                "explanation": "Text distortions or errors. Not considered artifacts in this task."
            },
            {
                "image_name": "clean_example4.png",
                "type": "Physical abnormality",
                "description": "A person walking past an overflowing trash can.",
                "explanation": "person walking towards the trash can. Not considered artifacts in this task."
            },
            {
                "image_name": "clean_example5.png",
                "type": "Physical abnormality",
                "description": "A bowl of food with gravy and a ladle sits on a table.",
                "explanation": "Spoon floating upon the gravy. Not considered in this scope."
            },
            {
                "image_name": "clean_example6.png",
                "type": "Cultural misalignment",
                "description": "Stop sign in front of a road with a car driving.",
                "explanation": "left-oriented road, Stop sign facing the wrong way. Not considered an artifact in this scope."
            }
        ]
    }
    
    return jsonify(examples)

@with_retry(max_retries=3, delay=0.2)
def load_prompts_from_disk():
    """Load prompts from disk with caching"""
    global _prompt_cache, _prompt_cache_time
    
    current_time = time.time()
    
    # Return cached data if still valid
    if _prompt_cache and (current_time - _prompt_cache_time) < CACHE_DURATION:
        logger.debug("Returning cached prompts")
        return _prompt_cache
    
    logger.info("Loading prompts from disk")
    
    # Look for JSON files in the images directory
    json_files = list(IMAGES_DIR.glob("*.json"))
    
    if not json_files:
        logger.warning("No JSON file found in images directory")
        return {"prompts": {}}
    
    # Use the first JSON file found
    prompts_file = json_files[0]
    logger.info(f"Using prompts file: {prompts_file}")
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a mapping from image names to prompts
        prompts_map = {}
        
        if 'images' in data and isinstance(data['images'], list):
            logger.info(f"Processing {len(data['images'])} image entries")
            
            # Get list of actual image files for efficient matching
            actual_images = {f.name for f in IMAGES_DIR.glob("*") 
                           if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}}
            
            for i, image_data in enumerate(data['images']):
                if not isinstance(image_data, dict):
                    continue
                
                caption_index = image_data.get('caption_index', i)
                prompt = image_data.get('prompt', '')
                
                if not prompt:  # Skip empty prompts
                    continue
                
                # Generate possible filenames for this prompt
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
                
                # Check which files actually exist
                found_match = False
                for name in possible_names:
                    if name in actual_images:
                        prompts_map[name] = prompt
                        logger.debug(f"Mapped {name} -> {prompt[:50]}...")
                        found_match = True
                        break
                
                if not found_match:
                    logger.debug(f"No match found for caption_index {caption_index}")
        
        result = {"prompts": prompts_map}
        
        # Update cache
        _prompt_cache = result
        _prompt_cache_time = current_time
        
        logger.info(f"Successfully loaded {len(prompts_map)} prompts")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in prompts file: {e}")
        return {"prompts": {}, "error": "Invalid JSON format"}
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        raise

@app.route('/api/prompts')
@ensure_session
@throttle_requests
def get_prompts():
    """Get prompt data for images with caching"""
    try:
        return jsonify(load_prompts_from_disk())
    except Exception as e:
        logger.error(f"Error in get_prompts: {e}")
        return jsonify({"prompts": {}, "error": "Failed to load prompts"}), 500

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

@app.route('/api/admin/cleanup-stale-assignments')
def cleanup_stale_assignments_route():
    """Admin endpoint to manually trigger cleanup of stale assignments"""
    try:
        cleaned_count = coordinator.cleanup_stale_assignments_manual()
        return jsonify({
            "success": True,
            "message": f"Cleaned up {cleaned_count} stale assignments",
            "cleaned_count": cleaned_count
        })
    except Exception as e:
        logger.error(f"Error during stale assignment cleanup: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/admin/assignment-status')
def get_assignment_status():
    """Admin endpoint to check current assignment status"""
    try:
        with coordinator.lock:
            current_time = time.time()
            status = {
                "classification_assignments": {},
                "annotation_assignments": {},
                "timeout_seconds": ASSIGNMENT_TIMEOUT
            }
            
            # Get classification assignments with ages
            for user_id, assignment in coordinator.progress["classification_in_progress"].items():
                if isinstance(assignment, dict):
                    age = int(current_time - assignment.get("timestamp", 0))
                    status["classification_assignments"][user_id] = {
                        "image": assignment.get("image"),
                        "age_seconds": age,
                        "is_stale": age > ASSIGNMENT_TIMEOUT
                    }
                else:
                    status["classification_assignments"][user_id] = {
                        "image": assignment,
                        "age_seconds": "unknown (old format)",
                        "is_stale": True
                    }
            
            # Get annotation assignments with ages
            for user_id, assignment in coordinator.progress["annotation_in_progress"].items():
                if isinstance(assignment, dict):
                    age = int(current_time - assignment.get("timestamp", 0))
                    status["annotation_assignments"][user_id] = {
                        "image": assignment.get("image"),
                        "age_seconds": age,
                        "is_stale": age > ASSIGNMENT_TIMEOUT
                    }
                else:
                    status["annotation_assignments"][user_id] = {
                        "image": assignment,
                        "age_seconds": "unknown (old format)",
                        "is_stale": True
                    }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting assignment status: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
