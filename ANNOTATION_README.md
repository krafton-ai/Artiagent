# Image Annotation System

A comprehensive web-based platform for collaborative image annotation with artifact detection and bounding box annotation.

## 🚀 Features

- **Two-Phase Annotation Process**:
  1. **Classification Phase**: Binary classification (artifact vs no-artifact)
  2. **Annotation Phase**: Bounding box drawing and detailed annotation for artifact images

- **Collaborative Work Management**:
  - Prevents duplicate work across multiple annotators
  - Real-time progress tracking
  - Automatic work assignment
  - Session management

- **Interactive Interface**:
  - Drag-and-drop bounding box creation with `[x1, y1, x2, y2]` format
  - Real-time canvas drawing with category-based color coding
  - Beautiful modal interface for annotation instead of browser popups
  - 4-way artifact category selection (Addition, Removal, Distortion, Fusion)
  - Comprehensive annotation guidelines with project-specific definitions
  - **Visual Examples**: Built-in examples showing proper annotation techniques
  - Progress visualization

- **Data Management**:
  - JSON-based result storage
  - Real-time data saving
  - Admin dashboard for monitoring
  - Export capabilities

## 📋 Requirements

- Python 3.7+
- Flask 2.3.3+
- Modern web browser with JavaScript enabled

## 🛠️ Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/minyoung.ahn/Workspace/image-artifacts
   ```

2. **Install dependencies**:
   ```bash
   pip install -r annotation_requirements.txt
   ```

3. **Create directories for your images**:
   ```bash
   mkdir annotation_images
   ```

4. **Add your images to the `annotation_images` directory**:
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Recommended size: Under 5MB per image

5. **Add prompt metadata (optional)**:
   - Place a JSON file in the `annotation_images` directory with prompt information
   - Follow the format: `{"metadata": {}, "images": [{"prompt": "...", "caption_index": 0}, ...]}`
   - Images will be matched to prompts using the `caption_index` field

## 🚀 Quick Start

1. **Start the server**:
   ```bash
   python annotation_server.py
   ```

2. **Access the system**:
   - Main interface: http://localhost:5000
   - Classification: http://localhost:5000/classify
   - Annotation: http://localhost:5000/annotate  
   - Admin Dashboard: http://localhost:5000/admin
   - Admin Classification (Unsure Images): http://localhost:5000/admin/classify
   - Admin Annotation (Admin-Classified Images): http://localhost:5000/admin/annotate
   - Upload Images: http://localhost:5000/upload

3. **Upload images** (optional - if not pre-loaded):
   - Go to http://localhost:5000/upload
   - Select and upload your image dataset

4. **Share links** with your annotators (public access):
   - Classification: http://localhost:5000/classify
   - Annotation: http://localhost:5000/annotate
   
5. **Admin-only interfaces** (not shared publicly):
   - Admin Dashboard: http://localhost:5000/admin
   - Admin Classification: http://localhost:5000/admin/classify
   - Admin Annotation: http://localhost:5000/admin/annotate

## 📊 Workflow

### Phase 1: Classification
1. Annotators visit the classification interface
2. System automatically assigns unique images to each annotator
3. Annotators classify each image as:
   - **"Has Artifacts" (True)**: Image contains visible artifacts
   - **"No Artifacts" (False)**: Image is clean and artifact-free
   - **"Unsure"**: Unclear/uncertain - reserved for admin review
4. Results are automatically saved
5. Progress is tracked in real-time

### Phase 2: Annotation
1. Only images marked as "Has Artifacts" proceed to annotation
2. Annotators draw bounding boxes around artifacts
3. Each bounding box gets assigned a category (Addition/Removal/Distortion/Fusion)
4. Each bounding box gets a detailed descriptive caption
5. Results are automatically saved

### Phase 3: Admin Review (For Unsure Images)
1. Images marked as "Unsure" are excluded from public annotation workflow
2. Admin accesses special classification interface at `/admin/classify`
3. Admin reviews uncertain images and makes final classification decision:
   - **"Has Artifacts"**: Image moves to admin annotation queue
   - **"No Artifacts"**: Image marked as clean and complete

### Phase 4: Admin Annotation (For Admin-Classified Images)
1. Images classified as "Has Artifacts" by admin become available for admin annotation
2. Admin accesses special annotation interface at `/admin/annotate`
3. Admin annotates all visible artifacts with same tools as regular annotation:
   - Draw bounding boxes around artifacts
   - Select category for each box (Addition/Removal/Distortion/Fusion)
   - Write detailed captions for each artifact
4. Admin annotations are stored with `task_type: "admin_annotation"`
5. All results stored in same dataset with proper admin tracking

## 📁 Data Structure

### Classification Results (`annotation_results/classification_results.json`)
```json
[
  {
    "image_name": "image001.jpg",
    "has_artifact": true,
    "user_id": "uuid-string",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "task_type": "classification"
  },
  {
    "image_name": "image002.jpg",
    "has_artifact": "unsure",
    "user_id": "uuid-string",
    "timestamp": "2024-01-15T10:32:00.000Z",
    "task_type": "classification"
  },
  {
    "image_name": "image002.jpg",
    "has_artifact": true,
    "user_id": "admin-uuid",
    "timestamp": "2024-01-15T10:45:00.000Z",
    "task_type": "admin_classification"
  }
]
```

**Classification Values:**
- `true`: Image contains artifacts
- `false`: Image has no artifacts  
- `"unsure"`: User uncertain - requires admin review

**Task Types:**
- `"classification"`: Regular user classification
- `"admin_classification"`: Admin review of unsure images

### Annotation Results (`annotation_results/annotation_results.json`)
```json
[
  {
    "image_name": "image001.jpg",
    "global_explanation": "",
    "bboxes": [
      {
        "bbox": [0.1, 0.2, 0.4, 0.6],
        "category": "distortion",
        "caption": "Color bleeding around subject edges with unnatural red/blue fringing effect"
      },
      {
        "bbox": [0.6, 0.7, 0.8, 0.8],
        "category": "addition", 
        "caption": "Extra background pixelation artifacts that create noise-like patterns"
      }
    ],
    "user_id": "uuid-string",
    "timestamp": "2024-01-15T10:35:00.000Z",
    "task_type": "annotation"
  },
  {
    "image_name": "image003.jpg",
    "global_explanation": "",
    "bboxes": [
      {
        "bbox": [0.15, 0.25, 0.45, 0.65],
        "category": "distortion",
        "caption": "Facial features are severely warped and asymmetrical, eyes different sizes"
      }
    ],
    "user_id": "admin-uuid",
    "timestamp": "2024-01-15T12:00:00.000Z",
    "task_type": "admin_annotation"
  }
]
```

**Task Types:**
- `"annotation"`: Regular user annotation  
- `"admin_annotation"`: Admin annotation of admin-classified images

**Bounding Box Format**: Each bbox entry contains:
- `bbox`: List of four floats `[x1, y1, x2, y2]` where:
  - `x1, y1`: Top-left corner coordinates
  - `x2, y2`: Bottom-right corner coordinates  
  - All coordinates are normalized (0-1 range) relative to image dimensions
- `category`: Artifact classification (addition/removal/distortion/fusion)
- `caption`: Detailed description of the specific artifact

**Categories**: Each bbox is classified into one of four categories:
- `addition`: Extra elements that shouldn't exist
- `removal`: Missing elements that should be present  
- `distortion`: Shape or appearance abnormalities
- `fusion`: Incorrectly merged or blended elements

### Prompt Metadata (`annotation_images/*.json`)
```json
{
  "metadata": {
    "description": "Image generation prompts for artifact annotation dataset",
    "created_date": "2024-01-15",
    "model": "flux-schnell",
    "total_images": 10
  },
  "images": [
    {
      "prompt": "A realistic portrait of a person with brown hair, wearing a blue shirt",
      "caption_index": 0
    },
    {
      "prompt": "A beautiful landscape with mountains and a lake",
      "caption_index": 1
    }
  ]
}
```

**Prompt Matching**: Images are matched to prompts using the `caption_index` field, which should correspond to the image filename pattern (e.g., `image_0000_flux-schnell.png` matches `caption_index: 0`).

## 👥 Multi-Annotator Coordination

The system automatically handles multiple simultaneous annotators:

- **Work Assignment**: Each annotator gets different images automatically
- **Progress Tracking**: Real-time updates prevent overlaps
- **Session Management**: Unique user sessions track individual progress
- **Conflict Prevention**: File locking prevents data corruption

## 🎯 Annotation Guidelines

### Artifact Classification
**What ARE artifacts**:
- Unnatural distortions or deformations
- Inconsistent lighting or shadows  
- Blurred or pixelated areas
- Text rendering issues
- Impossible physical structures
- Color bleeding or aberrations

**What are NOT artifacts**:
- Natural image noise or grain
- Motion blur from movement
- Depth of field effects
- Natural reflections or refractions
- Artistic stylistic choices
- Compression artifacts from file format

### Bounding Box Annotation
- **Drawing**: Click and drag to create boxes (stored as `[x1, y1, x2, y2]`)
- **Category Selection**: Choose from Addition, Removal, Distortion, or Fusion
- **Modal Interface**: Beautiful popup for category selection and caption writing
- **Color Coding**: Boxes are colored by category (Blue=Addition, Yellow=Removal, Red=Distortion, Green=Fusion)
- **Editing**: Double-click to edit categories and captions
- **Deletion**: Right-click to delete boxes
- **Guidelines**: Make boxes tight around artifacts, minimal background

### Visual Examples System
- **Built-in Examples**: Both classification and annotation interfaces include visual examples
- **Classification Examples**: Shows example images that should be marked as "Has Artifacts"
- **Annotation Examples**: Demonstrates proper bounding box placement and caption writing
- **Interactive Visualization**: Examples show color-coded bounding boxes with detailed annotations
- **Category-Specific**: Examples for each artifact type (Addition, Removal, Distortion, Fusion)
- **Real Annotations**: Examples use the same JSON format as actual annotations

### Prompt Display System
- **Generation Context**: Shows the original prompt used to generate each image
- **Metadata Integration**: Reads prompt data from JSON file in the annotation_images directory
- **Classification Support**: Prompts displayed under images during classification
- **Annotation Support**: Prompts displayed under images during annotation
- **Flexible Matching**: Automatically matches images to prompts using caption_index
- **Educational Value**: Helps annotators understand image generation context

## 📈 Admin Features

Access the admin dashboard at http://localhost:5000/admin to:

- Monitor real-time progress
- View completion statistics
- Download results (JSON format)
- Generate shareable links
- Upload additional images
- Export progress reports

## 🔧 Configuration

### Server Settings
Edit `annotation_server.py` to modify:
- Port (default: 5000)
- Host (default: 0.0.0.0 for external access)
- Directory paths
- Security settings

### Directory Structure
```
/annotation_images/          # Source images
/annotation_results/         # Result JSON files
  ├── classification_results.json
  ├── annotation_results.json  
  ├── work_assignments.json
  └── progress.json
/templates/                  # HTML templates
/static/                     # Static assets (CSS, JS)
```

## 🌐 Making the System Accessible Online

### Option 1: Local Network Access
The server runs on `0.0.0.0:5000` by default, making it accessible to other computers on your local network:
- Find your computer's IP address (e.g., 192.168.1.100)
- Share the link: `http://192.168.1.100:5000/classify`

### Option 2: Cloud Deployment
Deploy to cloud services like:
- **Heroku**: For easy deployment
- **AWS EC2**: For more control
- **Google Cloud**: For scalability
- **DigitalOcean**: For simplicity

### Option 3: Tunneling (Quick Setup)
Use tools like ngrok for quick external access:
```bash
# Install ngrok, then:
ngrok http 5000
# Share the generated https://xxx.ngrok.io URL
```

## 🔒 Security Considerations

For production deployment:
1. Change the Flask secret key
2. Enable HTTPS
3. Add authentication if needed
4. Validate file uploads more strictly
5. Set up proper backup procedures

## 📞 Support

### Common Issues

**Images not loading**: Check that images are in the `annotation_images` directory with supported formats.

**Multiple annotators getting same image**: Ensure system clock is synchronized and check for file permission issues.

**Data not saving**: Verify write permissions on the `annotation_results` directory.

**Server not accessible**: Check firewall settings and ensure port 5000 is open.

### Troubleshooting

1. **Check server logs**: The console output shows detailed error messages
2. **Verify directories**: Ensure all required directories exist and are writable
3. **Browser console**: Press F12 to check for JavaScript errors
4. **Network connectivity**: Test access from multiple devices

## 🎉 Getting Started Checklist

- [ ] Install Python and Flask
- [ ] Download/clone the annotation system
- [ ] Create `annotation_images` directory
- [ ] Add your image dataset
- [ ] Start the server with `python annotation_server.py`
- [ ] Test the system at http://localhost:5000
- [ ] Share classification link with annotators
- [ ] Monitor progress via admin dashboard
- [ ] Download results when complete

## 📊 Expected Performance

- **Classification**: ~10-30 seconds per image (depending on complexity)
- **Annotation**: ~2-5 minutes per artifact image
- **Concurrent Users**: Tested with 10-20 simultaneous annotators
- **Data Size**: Handles 1K+ images efficiently

The system is designed to be efficient and user-friendly while maintaining data integrity across multiple concurrent users.
