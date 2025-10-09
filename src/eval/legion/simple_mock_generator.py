"""
Simple mock LEGION response generator.

Since the LEGION source code isn't available, this creates realistic mock responses
with the correct format for testing the evaluation pipeline.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from pathlib import Path
import numpy as np
import pickle
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMockLegionGenerator:
    """Simple mock LEGION generator that creates realistic responses"""
    
    def __init__(self):
        logger.info("Initializing simple mock LEGION generator...")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Pre-defined artifact explanations for variety
        self.artifact_explanations = [
            "The image contains several structural artifacts including asymmetric facial features and distorted proportions in the central subject. There are also subtle color inconsistencies in the background lighting that suggest artificial generation.",
            
            "Physical artifacts are evident in the form of unnatural perspective distortions and violations of lighting physics. The shadows don't align properly with the light source, and there are texture inconsistencies in material surfaces.",
            
            "Multiple distortion artifacts are present including color banding in gradient areas, slight blurriness in fine details, and artificial-looking surface textures that appear overly smooth or synthetic.",
            
            "Structural deformities include asymmetric object proportions and anatomical inconsistencies. The image also shows signs of digital compression artifacts and color space conversion errors.",
            
            "The image exhibits physical law violations in lighting and shadow casting, along with material representation errors where surfaces appear to have inconsistent optical properties.",
            
            "Distortion artifacts include unnatural color saturation in certain regions, edge artifacts around object boundaries, and texture synthesis patterns that repeat unnaturally.",
            
            "Structural artifacts manifest as geometric inconsistencies and proportional errors in the main subjects. Background elements show signs of artificial pattern generation.",
            
            "Physical artifacts include impossible lighting scenarios and surface reflection inconsistencies. The image also contains subtle noise patterns characteristic of generative models.",
        ]
        
        logger.info("✅ Simple mock generator initialized")

    def _generate_realistic_heatmap(self, image: Image.Image) -> torch.Tensor:
        """Generate a realistic-looking heatmap based on image content"""
        width, height = image.size
        
        # Convert to numpy for processing
        img_array = np.array(image.convert('RGB'))
        
        # Create heatmap based on image features
        # Focus on edges, high contrast areas, and unusual patterns
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Simple edge detection
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))
        
        # Pad edges to match original size
        edges_x = np.pad(edges_x, ((0,0), (0,1)), mode='edge')
        edges_y = np.pad(edges_y, ((1,0), (0,0)), mode='edge')
        
        # Combine edges
        edges = edges_x + edges_y
        
        # Normalize and threshold
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        # Create artifact mask (focus on high edge areas)
        artifact_mask = (edges > 0.3).astype(float)
        
        # Add some random noise to make it more realistic
        noise = np.random.rand(height, width) * 0.2
        artifact_mask = np.clip(artifact_mask + noise, 0, 1)
        
        # Apply some morphological operations to create connected regions  
        try:
            from scipy import ndimage
            kernel = np.ones((5,5))
            artifact_mask = ndimage.binary_dilation(artifact_mask > 0.5, kernel).astype(float)
        except ImportError:
            # Fallback without scipy
            logger.warning("scipy not available, using simpler processing")
            artifact_mask = (artifact_mask > 0.5).astype(float)
        
        # Convert to torch tensor
        heatmap = torch.from_numpy(artifact_mask).int()
        
        return heatmap

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        """Generate mock response for single image"""
        try:
            # Generate realistic heatmap
            heatmap = self._generate_realistic_heatmap(image)
            
            # Select a random explanation
            explanation = np.random.choice(self.artifact_explanations)
            
            return {
                "heatmap": heatmap, 
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error in mock inference: {e}")
            # Return fallback response
            width, height = image.size
            return {
                "heatmap": torch.zeros((height, width), dtype=torch.int), 
                "explanation": "This image shows various types of visual artifacts commonly found in generated content.",
                "error": str(e)
            }


class SimpleDatasetIterator:
    """Simple iterator over dataset samples"""
    
    def __init__(self, dataset: str):
        self.dataset = dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset paths and data"""
        if self.dataset == 'synthscars':
            self.data_dir = Path("/data2/jhpark/image-artifacts/SynthScars/test")
            self.json_path = self.data_dir / "annotations" / "test.json"
        elif self.dataset == 'synartifact':
            self.data_dir = Path("/data2/jhpark/image-artifacts/SynArtifact/data")
            eval_set = self.data_dir / "eval.txt"
            if not eval_set.exists():
                logger.warning(f"Eval file not found: {eval_set}")
                self.data = []
                return
            self.data = []
            with open(eval_set, "r") as f:
                for line in f:
                    self.data.append(line.strip())
            logger.info(f"Loaded {len(self.data)} samples from {self.dataset}")
            return
        elif self.dataset == 'loki':
            self.data_dir = Path("/data2/jhpark/image-artifacts/loki")
            self.json_path = self.data_dir / "open_ended_vqa.json"
        elif self.dataset == 'richhf':
            self.data_dir = Path("/data2/jhpark/image-artifacts/richhf-18k") 
            self.json_path = self.data_dir / "test.json"
        elif self.dataset == 'ours':
            self.data_dir = Path("/data2/jhpark/image-artifacts/ours")
            self.json_path = self.data_dir / "metadata.json"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        # Check if files exist    
        if not self.json_path.exists():
            logger.warning(f"Annotations not found: {self.json_path}")
            # Create minimal mock data for testing
            self.data = []
            return
            
        # Handle different encodings for different datasets
        if self.dataset == 'loki':
            # LOKI JSON file is UTF-16 encoded
            with open(self.json_path, 'r', encoding='utf-16') as f:
                self.data = json.load(f)
        else:
            # Other datasets use UTF-8
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
        logger.info(f"Loaded {len(self.data)} samples from {self.dataset}")
    
    def _find_richhf_image_path(self, base_filename: str) -> Path:
        """
        Find the actual image path for RichHF dataset.
        Images are stored in numbered subdirectories but JSON doesn't specify which.
        
        Args:
            base_filename: Filename from JSON like "test/image.png"
            
        Returns:
            Path to actual image file
        """
        # Extract the image filename from the base path
        base_path = Path(base_filename)
        image_name = base_path.name  # e.g., "image.png"
        
        # Search in numbered subdirectories under test/
        test_dir = self.data_dir / "test"
        if test_dir.exists():
            # Try to find the image in any numbered subdirectory
            for subdir in test_dir.iterdir():
                if subdir.is_dir() and subdir.name.isdigit():
                    candidate_path = subdir / image_name
                    if candidate_path.exists():
                        return candidate_path
        
        # If not found, return original path as fallback
        return self.data_dir / base_filename
    
    def __iter__(self):
        # Handle different data structures for different datasets
        if self.dataset == 'richhf':
            # RichHF data is a nested dict: {id: {data}}, iterate over values
            iteration_data = self.data.values()
        else:
            # Other datasets are lists or dicts we iterate over directly
            iteration_data = self.data
            
        for item in iteration_data:
            try:
                if self.dataset == 'synthscars':
                    # item is a dict with image_id as key
                    image_id, json_data = next(iter(item.items()))
                    image_dir = self.data_dir / "images"
                    image_path = image_dir / json_data["img_file_name"]
                    
                elif self.dataset == 'synartifact':
                    # item is a path string like "root_folder/image.jpg"
                    root_folder = item.split('/')[0]
                    image_id = Path(item).stem
                    
                    image_path = self.data_dir / item
                    json_file = f"{root_folder}/annotation_json_artifacts_class/{image_id}.json"
                    json_path = self.data_dir / json_file
                    
                    with open(json_path, "r") as f:
                        json_data = json.load(f)
                        
                elif self.dataset == 'loki':
                    # item is json_data dict
                    json_data = item
                    image_path = self.data_dir / json_data["image_path"]
                    
                elif self.dataset == 'richhf':
                    # item is json_data dict
                    json_data = item
                    # RichHF images are in numbered subdirectories, but JSON doesn't specify which
                    # Need to search for the actual file location
                    base_filename = json_data["filename"]  # e.g., "test/image.png"
                    image_path = self._find_richhf_image_path(base_filename)
                elif self.dataset == 'ours':
                    # item is json_data dict
                    json_data = item
                    # Images are stored as {id}.png in images/ directory
                    image_path = self.data_dir / "images" / f"{json_data['id']}.png"
                else:
                    logger.warning(f"Unknown dataset: {self.dataset}")
                    continue
                    
                if image_path.exists():
                    yield json_data, image_path
                else:
                    logger.warning(f"Image not found: {image_path}")
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
    
    def __len__(self):
        return len(self.data)


def main():
    parser = argparse.ArgumentParser(description="Simple mock LEGION response generator")
    parser.add_argument('--datasets', nargs='+', default=['synthscars', 'synartifact', 'loki', 'richhf'],
                       help='Datasets to process')
    parser.add_argument('--output_dir', default='/data2/jhpark/image-artifacts/eval/legion_responses',
                       help='Output directory for mock responses')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum samples per dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize mock generator
    logger.info("Initializing simple mock LEGION generator...")
    generator = SimpleMockLegionGenerator()
    
    # Process each dataset
    for dataset in args.datasets:
        logger.info(f"🚀 Processing dataset: {dataset}")
        
        try:
            iterator = SimpleDatasetIterator(dataset)
            responses = {}
            
            # If no data available, create some dummy responses for testing
            if len(iterator) == 0:
                logger.info(f"No data found for {dataset}, creating dummy responses for testing...")
                # Create dummy responses
                for i in range(5):
                    dummy_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
                    response = generator.inference(dummy_image)
                    
                    # Convert torch tensors to numpy for serialization
                    if response.get("heatmap") is not None:
                        response["heatmap"] = response["heatmap"].numpy()
                    
                    responses[f"dummy_image_{i}.jpg"] = {
                        'response': response,
                        'json_data': {'image_path': f"dummy_image_{i}.jpg", 'has_artifact': True},
                        'timestamp': datetime.now().isoformat()
                    }
                continue
            
            total_samples = min(len(iterator), args.max_samples)
            logger.info(f"Processing {total_samples} samples from {dataset}")
            
            for i, (json_data, image_path) in enumerate(iterator):
                if i >= args.max_samples:
                    break
                    
                logger.info(f"Processing {i+1}/{total_samples}: {image_path.name}")
                
                try:
                    # Load image
                    image = Image.open(str(image_path)).convert("RGB")
                    if dataset == 'richhf':
                        image = image.resize((512, 512), Image.LANCZOS)
                    
                    # Generate mock response
                    response = generator.inference(image)
                    
                    # Convert torch tensors to numpy for serialization
                    if response.get("heatmap") is not None:
                        response["heatmap"] = response["heatmap"].numpy()
                    
                    # Store response with image path as key
                    responses[str(image_path.name)] = {
                        'response': response,
                        'json_data': json_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"✅ Completed {i+1}/{total_samples} samples")
                        
                except Exception as e:
                    logger.error(f"Error processing {image_path.name}: {e}")
                    responses[str(image_path.name)] = {
                        'response': {"heatmap": None, "explanation": "", "error": str(e)},
                        'json_data': json_data,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Save responses
            output_file = output_dir / f"{dataset}_responses.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(responses, f)
                
            logger.info(f"✅ Saved {len(responses)} mock responses to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            continue
    
    logger.info("🎉 Mock response generation completed!")
    logger.info("You can now test the evaluation pipeline with these mock responses.")


if __name__ == "__main__":
    main()
