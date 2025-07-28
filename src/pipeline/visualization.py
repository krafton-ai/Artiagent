import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import textwrap
import os
from typing import Union, Optional, List
from PIL import Image


class ImageVisualizer:
    """Handler for image visualization and display"""
    
    @staticmethod
    def _create_output_dir(image_name: str, base_dir: str = "output") -> str:
        """
        Create output directory for image-specific visualizations
        
        Args:
            image_name: Name of the image (used as directory name)
            base_dir: Base output directory
            
        Returns:
            Path to the created directory
        """
        # Remove file extension and create clean directory name
        clean_name = os.path.splitext(image_name)[0]
        output_dir = os.path.join(base_dir, clean_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def save_raw_image(image: Union[np.ndarray, Image.Image],
                       base_dir: str = "output",
                       filename: str = "raw_image.png"):
        """
        Save a raw PIL image without any matplotlib formatting
        
        Args:
            image: Image to save (numpy array or PIL Image)
            base_dir: Base output directory
            filename: Name of the output file
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Save the raw PIL image
        image.save(save_path)
        print(f"Raw image saved to {save_path}")
    
    @staticmethod
    def show_image(image: Union[np.ndarray, Image.Image], 
                   prompt: str = "",
                   figsize: tuple = (6, 6),
                   title: Optional[str] = None,
                   image_name: str = "unknown",
                   base_dir: str = "output",
                   filename: str = "image_output.png"):
        """
        Save a single image with optional caption
        
        Args:
            image: Image to display (numpy array or PIL Image)
            prompt: Caption/prompt text to display
            figsize: Figure size tuple (width, height)
            title: Optional title for the image
            image_name: Name of the image (used for directory creation)
            base_dir: Base output directory
            filename: Name of the output file
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Wrap caption text
        wrapped_caption = "\n".join(textwrap.wrap(prompt, width=80)) if prompt else ""
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Display image
        ax.imshow(image)
        ax.axis('off')
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add caption below the image
        if wrapped_caption:
            fig.text(0.5, 0.02, wrapped_caption, ha='center', va='bottom', 
                    fontsize=12, wrap=True)
            plt.tight_layout(rect=[0, 0.08, 1, 1])  # Adjust layout to fit caption
        else:
            plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    
    @staticmethod
    def show_comparison(original_image: Union[np.ndarray, Image.Image],
                       generated_image: Union[np.ndarray, Image.Image],
                       selected_instance_info: Optional[tuple] = None,
                       class_name: Optional[str] = None,
                       prompt: str = "",
                       figsize: tuple = (16, 8),
                       base_dir: str = "output",
                       filename: str = "comparison_output.png",
                       patch_data: Optional[dict] = None,
                       artifact_type: str = "addition",
                       kernel_type: str = "shuffle"):
        """
        Save original image with selected instance overlay and generated image side by side,
        with optional patch visualization based on artifact type
        
        Args:
            original_image: Original source image
            generated_image: Generated/modified image
            selected_instance_info: Selected instance info
            class_name: Class name of the selected instance
            prompt: Caption/prompt text to display
            figsize: Figure size tuple (width, height)
            image_name: Name of the image (used for directory creation)
            base_dir: Base output directory
            filename: Name of the output file
            patch_data: Dictionary containing reference_patch_indices and target_patch_indices
            artifact_type: Type of artifact ("addition", "removal", etc.)
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        
        # Convert PIL Images to numpy arrays if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        if isinstance(generated_image, Image.Image):
            generated_image = np.array(generated_image)
        

        # Wrap caption text
        wrapped_caption = "\n".join(textwrap.wrap(prompt, width=120)) if prompt else ""
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Display original image with selected instance overlay and patches
        axes[0].imshow(original_image)
        axes[0].axis('off')
        axes[0].set_title("Original", fontsize=14, fontweight='bold')
        
        # Draw patch visualization if patch_data is provided
        if patch_data:
            # Extract patch data
            reference_patches = patch_data.get('reference_patch_indices', []) or []
            target_patches = patch_data.get('target_patch_indices', []) or []
            
            # Convert patch indices (subtract 512 offset if present)
            reference_patches = [idx-512 for idx in reference_patches] if reference_patches else []
            target_patches = [idx-512 for idx in target_patches] if target_patches else []
            
            # Assume 16x16 patches for visualization (this could be made configurable)
            patch_size = 16
            h, w = original_image.shape[:2]
            patches_h = h // patch_size
            patches_w = w // patch_size
            
            # Choose which patches to display based on artifact type
            if artifact_type == "addition" and target_patches:
                patches_to_show = target_patches
                patch_color = 'blue'
                patch_label = 'Target Patches'
            else:
                patches_to_show = reference_patches
                patch_color = 'red'
                patch_label = 'Reference Patches'
            
            # Draw patch rectangles
            for i, patch_idx in enumerate(patches_to_show):
                row = patch_idx // patches_w
                col = patch_idx % patches_w
                # Only add label to first patch to avoid cluttering legend
                label = patch_label if i == 0 else None
                rect = patches.Rectangle((col * patch_size, row * patch_size), 
                                        patch_size, patch_size, 
                                        linewidth=2, edgecolor=patch_color, 
                                        facecolor=patch_color, alpha=0.3,
                                        label=label)
                axes[0].add_patch(rect)
        
        # Draw selected instance if provided
        if selected_instance_info:
            x1, y1, x2, y2 = selected_instance_info.get('bbox_coords', None)
            instance_rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor='yellow',
                facecolor='none',
                alpha=0.8,
                label=class_name
            )
            axes[0].add_patch(instance_rect)
        
        # Add legend if we have patches or selected instance
        if patch_data or selected_instance_info:
            axes[0].legend(loc='upper right', fontsize=10)
        
        # Display generated image
        axes[1].imshow(generated_image)
        axes[1].axis('off')
        axes[1].set_title('Generated', fontsize=14, fontweight='bold')
        
        # Add shared caption below the images
        if wrapped_caption:
            fig.text(0.5, 0.02, wrapped_caption, ha='center', va='bottom', 
                    fontsize=12, wrap=True)
            plt.tight_layout(rect=[0, 0.08, 1, 1])  # Adjust layout to fit caption
        else:
            plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    
    @staticmethod
    def show_detection_results(image: Union[np.ndarray, Image.Image],
                              visualized_output,
                              title: str = "Part Detection Results",
                              image_name: str = "unknown",
                              base_dir: str = "output",
                              filename: str = "detection_results.png"):
        """
        Save detection results from VLPart
        
        Args:
            image: Original image
            visualized_output: VLPart visualization output (can be matplotlib figure, PIL Image, or numpy array)
            title: Title for the visualization
            image_name: Name of the image (used for directory creation)
            base_dir: Base output directory
            filename: Name of the output file
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        
        if hasattr(visualized_output, 'fig'):
            # If visualized_output has a figure attribute, save it
            visualized_output.fig.suptitle(title, fontsize=14, fontweight='bold')
            visualized_output.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(visualized_output.fig)
            print(f"Plot saved to {save_path}")
        elif isinstance(visualized_output, Image.Image):
            # If visualized_output is a PIL Image, display it with matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(visualized_output)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {save_path}")
        elif isinstance(visualized_output, np.ndarray):
            # If visualized_output is a numpy array, display it with matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(visualized_output)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {save_path}")
        else:
            # Fallback to saving the original image
            ImageVisualizer.show_image(image, title=title, image_name=image_name, base_dir=base_dir, filename=filename)
    
    @staticmethod
    def show_bbox_overlay(image: Union[np.ndarray, Image.Image],
                         bbox: dict,
                         bbox_ref: Optional[dict] = None,
                         labels: Optional[List[str]] = None,
                         colors: Optional[List[str]] = None,
                         image_name: str = "unknown",
                         base_dir: str = "output",
                         filename: str = "bbox_overlay.png"):
        """
        Save image with bounding box overlays
        
        Args:
            image: Image to display
            bbox: Primary bounding box dict with xmin, ymin, xmax, ymax
            bbox_ref: Optional reference bounding box
            labels: Labels for the bounding boxes
            colors: Colors for the bounding boxes
            image_name: Name of the image (used for directory creation)
            base_dir: Base output directory
            filename: Name of the output file
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Default settings
        if labels is None:
            labels = ["Target", "Reference"] if bbox_ref else ["Target"]
        if colors is None:
            colors = ["red", "blue"] if bbox_ref else ["red"]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        
        # Draw primary bbox
        rect1 = plt.Rectangle(
            # (bbox['xmin'], bbox['ymin']),
            # bbox['xmax'] - bbox['xmin'],
            # bbox['ymax'] - bbox['ymin'],
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=colors[0],
            facecolor='none',
            label=labels[0]
        )
        ax.add_patch(rect1)
        
        # Draw reference bbox if provided
        if bbox_ref:
            rect2 = plt.Rectangle(
                (bbox_ref['xmin'], bbox_ref['ymin']),
                bbox_ref['xmax'] - bbox_ref['xmin'],
                bbox_ref['ymax'] - bbox_ref['ymin'],
                linewidth=2,
                edgecolor=colors[1],
                facecolor='none',
                label=labels[1]
            )
            ax.add_patch(rect2)
        
        ax.axis('off')
        # ax.legend(loc='upper right')
        # plt.title("Bounding Box Visualization", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    
    @staticmethod
    def show_artifact_comparison(real_image: Union[np.ndarray, Image.Image],
                                artifact_image: Union[np.ndarray, Image.Image],
                                reference_bbox: dict,
                                target_bbox: dict,
                                prompt: str = "",
                                titles: Optional[List[str]] = None,
                                figsize: tuple = (14, 7),
                                image_name: str = "unknown",
                                base_dir: str = "output",
                                filename: str = "artifact_comparison.png"):
        """
        Save side-by-side comparison of real image with reference bbox 
        and artifact injected image with target bbox
        
        Args:
            real_image: Original/real image
            artifact_image: Artifact injected image  
            reference_bbox: Reference bounding box dict with xmin, ymin, xmax, ymax
            target_bbox: Target bounding box dict with xmin, ymin, xmax, ymax
            prompt: Caption/prompt text to display
            titles: List of titles for [real, artifact] images
            figsize: Figure size tuple (width, height)
            image_name: Name of the image (used for directory creation)
            base_dir: Base output directory
            filename: Name of the output file
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        # Convert PIL Images to numpy arrays if needed
        if isinstance(real_image, Image.Image):
            real_image = np.array(real_image)
        if isinstance(artifact_image, Image.Image):
            artifact_image = np.array(artifact_image)
        
        # Default titles
        if titles is None:
            titles = ["Real Image + Reference BBox", "Artifact Injected + Target BBox"]
        
        # Wrap caption text
        wrapped_caption = "\n".join(textwrap.wrap(prompt, width=100)) if prompt else ""
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Display real image with reference bbox
        axes[0].imshow(real_image)
        axes[0].axis('off')
        axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
        
        # Draw reference bbox
        ref_rect = plt.Rectangle(
            (reference_bbox['xmin'], reference_bbox['ymin']),
            reference_bbox['xmax'] - reference_bbox['xmin'],
            reference_bbox['ymax'] - reference_bbox['ymin'],
            linewidth=3,
            edgecolor='blue',
            facecolor='none',
            label='Reference'
        )
        axes[0].add_patch(ref_rect)
        axes[0].legend(loc='upper right')
        
        # Display artifact image with target bbox
        axes[1].imshow(artifact_image)
        axes[1].axis('off')
        axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
        
        # Draw target bbox
        target_rect = plt.Rectangle(
            (target_bbox['xmin'], target_bbox['ymin']),
            target_bbox['xmax'] - target_bbox['xmin'],
            target_bbox['ymax'] - target_bbox['ymin'],
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            label='Target'
        )
        axes[1].add_patch(target_rect)
        axes[1].legend(loc='upper right')
        
        # Add shared caption below the images
        if wrapped_caption:
            fig.text(0.5, 0.02, wrapped_caption, ha='center', va='bottom', 
                    fontsize=12, wrap=True)
            plt.tight_layout(rect=[0, 0.08, 1, 1])  # Adjust layout to fit caption
        else:
            plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    
    @staticmethod
    def save_image(image: Union[np.ndarray, Image.Image], 
                   filepath: str,
                   format: str = 'PNG'):
        """
        Save image to file
        
        Args:
            image: Image to save
            filepath: Output file path
            format: Image format (PNG, JPEG, etc.)
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        pil_image.save(filepath, format=format)
        print(f"Image saved to {filepath}")
    
    @staticmethod
    def create_grid(images: List[Union[np.ndarray, Image.Image]],
                   titles: Optional[List[str]] = None,
                   grid_size: Optional[tuple] = None,
                   figsize: tuple = (12, 8),
                   image_name: str = "unknown",
                   base_dir: str = "output",
                   filename: str = "image_grid.png"):
        """
        Create and save a grid of images
        
        Args:
            images: List of images to display
            titles: Optional list of titles for each image
            grid_size: Tuple (rows, cols) for grid layout. Auto-calculated if None
            figsize: Figure size tuple
            image_name: Name of the image (used for directory creation)
            base_dir: Base output directory
            filename: Name of the output file
        """
        # Use base_dir directly (no additional subdirectory creation)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, filename)
        n_images = len(images)
        
        # Auto-calculate grid size if not provided
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle single image case
        if n_images == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, image in enumerate(images):
            if i >= len(axes):
                break
                
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            axes[i].imshow(image)
            axes[i].axis('off')
            
            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=12)
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}") 