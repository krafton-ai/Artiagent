import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import argparse
from typing import Tuple, Dict, List
import random
import pickle

class DINOImageComparator:
    """Class for comparing images using DINO embeddings"""
    
    def __init__(self, model_name: str = 'facebook/dinov2-base'):
        """
        Initialize DINO model and processor
        
        Args:
            model_name: HuggingFace model name for DINO
        """
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.patch_size = self.model.config.patch_size
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Loaded DINO model: {model_name}")
        print(f"Patch size: {self.patch_size}")
        print("Running on CPU")
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate image from path"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path} ({image.width}x{image.height})")
        return image
    
    def extract_embeddings(self, image: Image.Image, mask: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DINO embeddings from image
        
        Args:
            image: PIL Image to process
            mask: Optional binary mask (H, W) to select specific patches
            
        Returns:
            Tuple of (cls_embedding, patch_embeddings)
        """
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Get image dimensions
        batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
        num_patches_height = img_height // self.patch_size
        num_patches_width = img_width // self.patch_size
        num_patches_flat = num_patches_height * num_patches_width
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs[0]
        
        # Separate CLS token and patch embeddings
        cls_token = last_hidden_states[:, 0, :]  # [1, hidden_size]
        patch_features = last_hidden_states[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))
        
        # If mask is provided, filter patches
        if mask is not None:
            # Resize mask to patch grid
            mask_resized = self._resize_mask_to_patches(mask, num_patches_height, num_patches_width)
            # Get masked patch indices
            masked_patches = self._get_masked_patches(patch_features.squeeze(0), mask_resized)
            return cls_token.squeeze(0), masked_patches
        
        return cls_token.squeeze(0), patch_features.squeeze(0)  # Remove batch dimension
    
    def _resize_mask_to_patches(self, mask: np.ndarray, num_patches_height: int, num_patches_width: int) -> np.ndarray:
        """Resize mask to patch grid dimensions"""
        from PIL import Image
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_resized = mask_pil.resize((num_patches_width, num_patches_height), Image.NEAREST)
        return np.array(mask_resized) > 0
    
    def _get_masked_patches(self, patch_features: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """Extract patches where mask is True"""
        # Convert mask to boolean tensor
        mask_tensor = torch.from_numpy(mask).bool()
        
        # Get indices where mask is True
        masked_indices = torch.where(mask_tensor.flatten())[0]
        
        # Extract masked patches
        patch_features_flat = patch_features.flatten(0, 1)  # [num_patches, hidden_size]
        masked_patches = patch_features_flat[masked_indices]
        
        return masked_patches
    
    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings using L2 norm"""
        return F.normalize(embeddings, p=2, dim=-1)
    
    def compute_cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two normalized embeddings"""
        # Ensure embeddings are normalized
        emb1_norm = self.normalize_embeddings(emb1)
        emb2_norm = self.normalize_embeddings(emb2)
        
        # Compute cosine similarity
        # If embeddings are 1D, compute directly
        if emb1_norm.dim() == 1 and emb2_norm.dim() == 1:
            similarity = F.cosine_similarity(emb1_norm.unsqueeze(0), emb2_norm.unsqueeze(0), dim=1)
        else:
            # For multi-dimensional embeddings, flatten and compute
            emb1_flat = emb1_norm.flatten()
            emb2_flat = emb2_norm.flatten()
            similarity = F.cosine_similarity(emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0), dim=1)
        
        return similarity.item()
    
    def classify_similarity(self, similarity: float, thresholds: Dict[str, float]) -> Tuple[str, str]:
        """
        Classify similarity based on cosine similarity score
        
        Args:
            similarity: Cosine similarity score (-1 to 1)
            thresholds: Dictionary with 'same', 'similar', 'different' thresholds
            
        Returns:
            Tuple of (classification, reasoning)
        """
        if similarity >= thresholds['same']:
            classification = "same"
            reasoning = f"Very high similarity ({similarity:.3f} >= {thresholds['same']}) indicates identical or nearly identical images"
        elif similarity >= thresholds['similar']:
            classification = "similar"
            reasoning = f"Moderate similarity ({similarity:.3f} >= {thresholds['similar']}) suggests same content with distortions or variations"
        else:
            classification = "different"
            reasoning = f"Low similarity ({similarity:.3f} < {thresholds['similar']}) indicates different content"
        
        return classification, reasoning
    
    def visualize_comparison(self, img1: Image.Image, img2: Image.Image, 
                           similarity: float, classification: str, reasoning: str,
                           output_path: str = None,
                           mask1: np.ndarray = None,
                           mask2: np.ndarray = None,
                           patch_indices1: List[int] = None,
                           patch_indices2: List[int] = None):
        """Create visualization of image comparison with patch selection overlay"""
        # Create subplot layout based on whether we have patch information
        has_patch_info = (mask1 is not None or mask2 is not None or 
                         patch_indices1 is not None or patch_indices2 is not None)
        
        if has_patch_info:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            # Flatten axes for easier indexing
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot first image
        axes[0].imshow(img1)
        axes[0].set_title(f"Image 1\n{img1.width}x{img1.height}")
        
        # Overlay mask or patch selection on first image
        if mask1 is not None:
            # Resize mask to image dimensions
            mask1_resized = self._resize_mask_to_image(mask1, img1.size)
            axes[0].imshow(mask1_resized, alpha=0.3, cmap='Reds')
            axes[0].set_title(f"Image 1 (with mask)\n{img1.width}x{img1.height}")
        elif patch_indices1 is not None:
            self._overlay_patch_indices(axes[0], img1, patch_indices1, color='red')
            axes[0].set_title(f"Image 1 (selected patches)\n{img1.width}x{img1.height}")
        
        axes[0].axis('off')
        
        # Plot second image
        axes[1].imshow(img2)
        axes[1].set_title(f"Image 2\n{img2.width}x{img2.height}")
        
        # Overlay mask or patch selection on second image
        if mask2 is not None:
            # Resize mask to image dimensions
            mask2_resized = self._resize_mask_to_image(mask2, img2.size)
            axes[1].imshow(mask2_resized, alpha=0.3, cmap='Blues')
            axes[1].set_title(f"Image 2 (with mask)\n{img2.width}x{img2.height}")
        elif patch_indices2 is not None:
            self._overlay_patch_indices(axes[1], img2, patch_indices2, color='blue')
            axes[1].set_title(f"Image 2 (selected patches)\n{img2.width}x{img2.height}")
        
        axes[1].axis('off')
        
        # Plot similarity info
        axes[2].text(0.1, 0.8, f"Cosine Similarity: {similarity:.3f}", 
                    fontsize=14, fontweight='bold')
        axes[2].text(0.1, 0.6, f"Classification: {classification.upper()}", 
                    fontsize=12, fontweight='bold')
        axes[2].text(0.1, 0.4, "Reasoning:", fontsize=10, fontweight='bold')
        
        # Wrap reasoning text
        words = reasoning.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) < 40:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for i, line in enumerate(lines):
            axes[2].text(0.1, 0.3 - i*0.05, line, fontsize=9)
        
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        # Add patch grid visualization if we have patch information
        if has_patch_info:
            # Patch grid for image 1
            axes[3].imshow(img1)
            self._draw_patch_grid(axes[3], img1, self.patch_size, alpha=0.7)
            if patch_indices1 is not None:
                self._highlight_patches(axes[3], img1, patch_indices1, self.patch_size, color='red')
            axes[3].set_title("Image 1 - Patch Grid")
            axes[3].axis('off')
            
            # Patch grid for image 2
            axes[4].imshow(img2)
            self._draw_patch_grid(axes[4], img2, self.patch_size, alpha=0.7)
            if patch_indices2 is not None:
                self._highlight_patches(axes[4], img2, patch_indices2, self.patch_size, color='blue')
            axes[4].set_title("Image 2 - Patch Grid")
            axes[4].axis('off')
            
            # Patch statistics
            axes[5].text(0.1, 0.9, "Patch Statistics", fontsize=14, fontweight='bold')
            if patch_indices1 is not None:
                axes[5].text(0.1, 0.8, f"Image 1 patches: {len(patch_indices1)}", fontsize=12)
            if patch_indices2 is not None:
                axes[5].text(0.1, 0.7, f"Image 2 patches: {len(patch_indices2)}", fontsize=12)
            if mask1 is not None:
                axes[5].text(0.1, 0.6, f"Image 1 mask: {mask1.shape}", fontsize=12)
            if mask2 is not None:
                axes[5].text(0.1, 0.5, f"Image 2 mask: {mask2.shape}", fontsize=12)
            axes[5].set_xlim(0, 1)
            axes[5].set_ylim(0, 1)
            axes[5].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        
        plt.show()
    
    def _resize_mask_to_image(self, mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """Resize mask to image dimensions"""
        from PIL import Image
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_resized = mask_pil.resize(image_size, Image.NEAREST)
        return np.array(mask_resized) > 0
    
    def _overlay_patch_indices(self, ax, image: Image.Image, patch_indices: List[int], color: str = 'red'):
        """Overlay patch indices on image"""
        # Convert patch indices to coordinates
        img_width, img_height = image.size
        num_patches_width = img_width // self.patch_size
        num_patches_height = img_height // self.patch_size
        
        for patch_idx in patch_indices:
            # Convert flat index to 2D coordinates
            patch_y = patch_idx // num_patches_width
            patch_x = patch_idx % num_patches_width
            
            # Convert to pixel coordinates
            x = patch_x * self.patch_size
            y = patch_y * self.patch_size
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), self.patch_size, self.patch_size, 
                               linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
    
    def _draw_patch_grid(self, ax, image: Image.Image, patch_size: int, alpha: float = 0.7):
        """Draw patch grid on image"""
        img_width, img_height = image.size
        num_patches_width = img_width // patch_size
        num_patches_height = img_height // patch_size
        
        # Draw vertical lines
        for i in range(num_patches_width + 1):
            x = i * patch_size
            ax.axvline(x=x, color='white', alpha=alpha, linewidth=1)
        
        # Draw horizontal lines
        for i in range(num_patches_height + 1):
            y = i * patch_size
            ax.axhline(y=y, color='white', alpha=alpha, linewidth=1)
    
    def _highlight_patches(self, ax, image: Image.Image, patch_indices: List[int], 
                          patch_size: int, color: str = 'red'):
        """Highlight specific patches on the grid"""
        img_width, img_height = image.size
        num_patches_width = img_width // patch_size
        
        for patch_idx in patch_indices:
            # Convert flat index to 2D coordinates
            patch_y = patch_idx // num_patches_width
            patch_x = patch_idx % num_patches_width
            
            # Convert to pixel coordinates
            x = patch_x * patch_size
            y = patch_y * patch_size
            
            # Draw filled rectangle
            rect = plt.Rectangle((x, y), patch_size, patch_size, 
                               linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
    
    def compare_images(self, img1_path: str, img2_path: str, 
                      thresholds: Dict[str, float] = None,
                      output_path: str = None,
                      mask1: np.ndarray = None,
                      mask2: np.ndarray = None) -> Dict:
        """
        Compare two images and return similarity analysis
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            thresholds: Similarity thresholds for classification
            output_path: Path to save visualization
            
        Returns:
            Dictionary with comparison results
        """
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = {
                'same': 0.95,      # Very high similarity
                'similar': 0.7,    # Moderate similarity
                'different': 0.0   # Low similarity
            }
        
        # Load images
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)
        
        # Extract embeddings
        print("Extracting embeddings...")
        if mask1 is not None:
            print(f"Using mask for image 1: {mask1.shape}")
        if mask2 is not None:
            print(f"Using mask for image 2: {mask2.shape}")
            
        cls1, patches1 = self.extract_embeddings(img1, mask1)
        cls2, patches2 = self.extract_embeddings(img2, mask2)
        
        # Compute similarities
        cls_similarity = self.compute_cosine_similarity(cls1, cls2)
        
        # Flatten patch embeddings for comparison
        patches1_flat = patches1.flatten(0, 1)  # [num_patches, hidden_size]
        patches2_flat = patches2.flatten(0, 1)  # [num_patches, hidden_size]
        
        # Use minimum number of patches for comparison
        min_patches = min(patches1_flat.shape[0], patches2_flat.shape[0])
        patches1_flat = patches1_flat[:min_patches]
        patches2_flat = patches2_flat[:min_patches]
        
        patch_similarity = self.compute_cosine_similarity(patches1_flat, patches2_flat)
        
        # Use average of CLS and patch similarities
        avg_similarity = (cls_similarity + patch_similarity) / 2
        
        # Classify similarity
        classification, reasoning = self.classify_similarity(avg_similarity, thresholds)
        
        # Create results dictionary
        results = {
            'cls_similarity': cls_similarity,
            'patch_similarity': patch_similarity,
            'avg_similarity': avg_similarity,
            'classification': classification,
            'reasoning': reasoning,
            'thresholds': thresholds
        }
        
        # Print results
        print("\n" + "="*50)
        print("IMAGE SIMILARITY ANALYSIS")
        print("="*50)
        print(f"CLS Token Similarity: {cls_similarity:.3f}")
        print(f"Patch Similarity: {patch_similarity:.3f}")
        print(f"Average Similarity: {avg_similarity:.3f}")
        print(f"Classification: {classification.upper()}")
        print(f"Reasoning: {reasoning}")
        print("="*50)
        
        # Create visualization with patch information
        self.visualize_comparison(img1, img2, avg_similarity, classification, reasoning, 
                                 output_path, mask1, mask2)
        
        return results

    def compare_with_patch_indices(self, img1_path: str, img2_path: str,
                                  patch_indices1: List[int] = None,
                                  patch_indices2: List[int] = None,
                                  thresholds: Dict[str, float] = None,
                                  output_path: str = None) -> Dict:
        """
        Compare images using specific patch indices
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            patch_indices1: List of patch indices to use from first image
            patch_indices2: List of patch indices to use from second image
            thresholds: Similarity thresholds for classification
            output_path: Path to save visualization
            
        Returns:
            Dictionary with comparison results
        """
        # Load images
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)
        
        # Extract full embeddings first
        print("Extracting full embeddings...")
        cls1, patches1_full = self.extract_embeddings(img1)
        cls2, patches2_full = self.extract_embeddings(img2)
        
        # Filter patches by indices if provided
        if patch_indices1 is not None:
            print(f"Using {len(patch_indices1)} patch indices for image 1")
            patches1 = patches1_full.flatten(0, 1)[patch_indices1]
        else:
            patches1 = patches1_full.flatten(0, 1)
            
        if patch_indices2 is not None:
            print(f"Using {len(patch_indices2)} patch indices for image 2")
            patches2 = patches2_full.flatten(0, 1)[patch_indices2]
        else:
            patches2 = patches2_full.flatten(0, 1)
        
        # Compute similarities
        cls_similarity = self.compute_cosine_similarity(cls1, cls2)
        patch_similarity = self.compute_cosine_similarity(patches1, patches2)
        
        # Use average of CLS and patch similarities
        avg_similarity = (cls_similarity + patch_similarity) / 2
        
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = {
                'same': 0.95,      # Very high similarity
                'similar': 0.7,    # Moderate similarity
                'different': 0.0   # Low similarity
            }
        
        # Classify similarity
        classification, reasoning = self.classify_similarity(avg_similarity, thresholds)
        
        # Create results dictionary
        results = {
            'cls_similarity': cls_similarity,
            'patch_similarity': patch_similarity,
            'avg_similarity': avg_similarity,
            'classification': classification,
            'reasoning': reasoning,
            'thresholds': thresholds,
            'patch_indices1': patch_indices1,
            'patch_indices2': patch_indices2
        }
        
        # Print results
        print("\n" + "="*50)
        print("PATCH-BASED IMAGE SIMILARITY ANALYSIS")
        print("="*50)
        print(f"CLS Token Similarity: {cls_similarity:.3f}")
        print(f"Patch Similarity: {patch_similarity:.3f}")
        print(f"Average Similarity: {avg_similarity:.3f}")
        print(f"Classification: {classification.upper()}")
        print(f"Reasoning: {reasoning}")
        if patch_indices1:
            print(f"Patches used (Image 1): {len(patch_indices1)}")
        if patch_indices2:
            print(f"Patches used (Image 2): {len(patch_indices2)}")
        print("="*50)
        
        # Create visualization with patch information
        self.visualize_comparison(img1, img2, avg_similarity, classification, reasoning, 
                                 output_path, patch_indices1=patch_indices1, patch_indices2=patch_indices2)
        
        return results


def get_random_images_from_directory(directory: str, num_images: int = 2) -> List[str]:
    """Get random image paths from directory"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, filename))
    
    if len(image_files) < num_images:
        raise ValueError(f"Directory contains only {len(image_files)} images, need {num_images}")
    
    # Randomly select images
    selected_images = random.sample(image_files, num_images)
    return selected_images



def main():
    """Main function for image comparison"""
    parser = argparse.ArgumentParser(description='Compare images using DINO embeddings')
    parser.add_argument('--data-dir', type=str, default='data/coco_2017_extracted/train2017',
                       help='Directory containing images')
    parser.add_argument('--imgid', type=int)
    parser.add_argument('--same-threshold', type=float, default=0.95,
                       help='Threshold for "same" classification')
    parser.add_argument('--similar-threshold', type=float, default=0.7,
                       help='Threshold for "similar" classification')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = DINOImageComparator()
    
    # Get image paths
    img1_path = f"flux_output_coco_animal/image_{args.imgid}/01_original_image.png"
    img2_path = f"flux_output_coco_animal/image_{args.imgid}/artifact_distortion.png"

    if args.output == None:
        output_path = f"../outputs/filter/filtered_{args.imgid}"
    else:
        output_path = args.output

    # Obtain masks from path
    metadata_path = f"gsam_output_coco_animal/image_{args.imgid}/metadata.pkl"

    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)

    # Set thresholds
    thresholds = {
        'same': args.same_threshold,
        'similar': args.similar_threshold,
        'different': 0.0
    }
    
    # Compare images
    results = comparator.compare_images(
        img1_path, img2_path, 
        thresholds=thresholds,
        mask1=data['artifacts']['distortion']['masks'],
        mask2=data['artifacts']['distortion']['masks'],
        output_path=output_path
    )
    
    return results




if __name__ == "__main__":
    main()