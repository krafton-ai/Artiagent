"""
Evaluation utilities for artifact detection models.

This module provides evaluation metrics and utilities for measuring
the performance of VLM/MLLM models on artifact detection tasks.
"""

import numpy as np
import json
import re
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
try:
    from shapely.geometry import Polygon, box
except ImportError:
    Polygon = box = None
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = util = None
try:
    import tensorflow as tf
except ImportError:
    tf = None

def parse_tfrecord_file(filename: str) -> List[Dict]:
    """
    Parse a TFRecord file from RichHF-18K dataset.
    
    Based on https://github.com/google-research/google-research/blob/master/richhf_18k/parse_tfrecord_file.py
    
    Args:
        filename: Path to the TFRecord file
        
    Returns:
        List of parsed samples with artifact information
    """
    if tf is None:
        raise ImportError("TensorFlow is required for parsing TFRecord files. Install with: pip install tensorflow")
    
    samples = []
    raw_dataset = tf.data.TFRecordDataset(filename)
    
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        feat_map = example.features.feature

        # Extract features according to RichHF-18K format
        sample = {
            # Original filename which can be mapped to images in pick-a-pic dataset
            'filename': feat_map['filename'].bytes_list.value[0].decode(),
            
            # 4 fine-grained scores
            'aesthetics_score': feat_map['aesthetics_score'].float_list.value[0],
            'artifact_score': feat_map['artifact_score'].float_list.value[0],
            'misalignment_score': feat_map['misalignment_score'].float_list.value[0],
            'overall_score': feat_map['overall_score'].float_list.value[0],
            
            # Artifact and misalignment heatmaps
            'artifact_map': tf.image.decode_image(
                feat_map['artifact_map'].bytes_list.value[0], channels=1
            ).numpy(),
            'misalignment_map': tf.image.decode_image(
                feat_map['misalignment_map'].bytes_list.value[0], channels=1
            ).numpy(),
            
            # Misalignment label for token mapping
            'prompt_misalignment_label': feat_map['prompt_misalignment_label'].bytes_list.value[0].decode()
        }
        
        samples.append(sample)
    
    return samples

class Visualizer:
    """
    Visualization utilities for artifact detection results.
    
    This class provides methods to visualize predictions vs ground truth
    for different dataset types with proper labeling and saving capabilities.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load a default font for text rendering
        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
            self.title_font = ImageFont.truetype("arial.ttf", 28)
        except:
            try:
                # Fallback to default font
                self.font = ImageFont.load_default()
                self.title_font = ImageFont.load_default()
            except:
                self.font = None
                self.title_font = None
    
    def _draw_bbox(self, draw: ImageDraw.Draw, bbox: List[float], 
                   color: Tuple[int, int, int, int], width: int = 3, 
                   label: str = None):
        """
        Draw a bounding box on the image.
        
        Args:
            draw: ImageDraw object
            bbox: Bounding box coordinates [x1, y1, x2, y2] or [x, y, w, h]
            color: RGBA color tuple
            width: Line width
            label: Optional label text
        """
        if len(bbox) == 4:
            # Handle both [x1,y1,x2,y2] and [x,y,w,h] formats
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                # Assume [x1,y1,x2,y2] format
                x1, y1, x2, y2 = bbox
            else:
                # Assume [x,y,w,h] format
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            # Add label if provided
            # if label and self.font:
            #     text_bbox = draw.textbbox((x1, y1-25), label, font=self.font)
            #     draw.rectangle(text_bbox, fill=color)
            #     draw.text((x1, y1-25), label, fill=(255, 255, 255, 255), font=self.font)
    
    def _draw_polygon(self, draw: ImageDraw.Draw, polygon: List[float], 
                      color: Tuple[int, int, int, int], label: str = None):
        """
        Draw a polygon on the image.
        
        Args:
            draw: ImageDraw object
            polygon: Polygon coordinates as flat list [x1, y1, x2, y2, ..., xn, yn]
            color: RGBA color tuple
            label: Optional label text
        """
        if len(polygon) >= 6:  # At least 3 points
            # Convert flat coordinates to point pairs
            points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            
            # Draw polygon outline
            draw.polygon(points, outline=color, width=3)
            
            # Add label if provided
            # if label and self.font and points:
            #     x, y = points[0]  # Use first point for label position
            #     text_bbox = draw.textbbox((x, y-25), label, font=self.font)
            #     draw.rectangle(text_bbox, fill=color)
            #     draw.text((x, y-25), label, fill=(255, 255, 255, 255), font=self.font)
    
    def visualize_loki(self, image: Image.Image, result: Dict, sample: Dict, 
                       sample_idx: int, save_path: str = None) -> str:
        """
        Visualize LOKI dataset results with predictions and ground truth.
        
        Args:
            image: PIL Image
            result: Model prediction results
            sample: Ground truth sample data
            sample_idx: Sample index for naming
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization
        """
        # Create a copy for visualization
        image_viz = image.convert("RGBA")
        draw = ImageDraw.Draw(image_viz)
        
        # Draw predicted bounding boxes (red)
        num_artifacts = result.get('number_of_artifacts', 0)
        if num_artifacts > 0 and 'artifacts' in result:
            for i, artifact in enumerate(result['artifacts']):
                bbox = artifact.get('bbox_2d', [])
                explanation = artifact.get('explanation', f'Pred {i+1}')
                
                if len(bbox) == 4:
                    self._draw_bbox(draw, bbox, (255, 0, 0, 180), 
                                  label=f"PRED: {explanation[:30]}")
        
        # Draw ground truth bounding boxes (green)
        if 'problems' in sample and 'regional' in sample['problems']:
            for i, region_info in enumerate(sample['problems']['regional']):
                if 'region' in region_info:
                    x, y, w, h = region_info['region']
                    bbox = [x, y, x + w, y + h]  # Convert to [x1,y1,x2,y2]
                    desc = region_info.get('desc', f'GT {i+1}')
                    
                    self._draw_bbox(draw, bbox, (0, 255, 0, 180), 
                                  label=f"GT: {desc[:30]}")
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / f"loki_sample_{sample_idx:04d}.png"
        else:
            save_path = Path(save_path)
        
        # Convert back to RGB for saving
        image_rgb = Image.new("RGB", image_viz.size, (255, 255, 255))
        image_rgb.paste(image_viz, mask=image_viz.split()[-1])
        image_rgb.save(save_path, "PNG")
        
        return str(save_path)
    
    def visualize_synartifact(self, image: Image.Image, result: Dict, sample: Dict, 
                              sample_idx: int, save_path: str = None) -> str:
        """
        Visualize SynArtifact dataset results with predictions and ground truth.
        
        Args:
            image: PIL Image
            result: Model prediction results
            sample: Ground truth sample data
            sample_idx: Sample index for naming
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization
        """
        # Create a copy for visualization
        image_viz = image.convert("RGBA")
        draw = ImageDraw.Draw(image_viz)
        
        # Draw predicted bounding boxes (red)
        num_artifacts = result.get('number_of_artifacts', 0)
        if num_artifacts > 0 and 'artifacts' in result:
            for i, artifact in enumerate(result['artifacts']):
                bbox = artifact.get('bbox_2d', [])
                explanation = artifact.get('explanation', f'Pred {i+1}')
                
                if len(bbox) == 4:
                    self._draw_bbox(draw, bbox, (255, 0, 0, 180), 
                                  label=f"PRED: {explanation[:30]}")
        
        # Draw ground truth bounding boxes (green)
        annotations = sample.get('Artifacts annotation', [])
        if annotations:
            for i, annotation in enumerate(annotations):
                if 'rect_start' in annotation and 'rect_end' in annotation:
                    x1, y1 = annotation['rect_start']
                    x2, y2 = annotation['rect_end']
                    bbox = [x1, y1, x2, y2]
                    desc = annotation.get('artifacts_caption', f'GT {i+1}')
                    
                    self._draw_bbox(draw, bbox, (0, 255, 0, 180), 
                                  label=f"GT: {desc[:30]}")
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / f"synartifact_sample_{sample_idx:04d}.png"
        else:
            save_path = Path(save_path)
        
        # Convert back to RGB for saving
        image_rgb = Image.new("RGB", image_viz.size, (255, 255, 255))
        image_rgb.paste(image_viz, mask=image_viz.split()[-1])
        image_rgb.save(save_path, "PNG")
        
        return str(save_path)
    
    def visualize_synthscars(self, image: Image.Image, result: Dict, sample: Dict, 
                             sample_idx: int, save_path: str = None) -> str:
        """
        Visualize SynthScars dataset results with predictions and ground truth polygons.
        
        Args:
            image: PIL Image
            result: Model prediction results
            sample: Ground truth sample data
            sample_idx: Sample index for naming
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization
        """
        # Use matplotlib for better polygon visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw predicted bounding boxes (red rectangles)
        num_artifacts = result.get('number_of_artifacts', 0)
        if num_artifacts > 0 and 'artifacts' in result:
            for i, artifact in enumerate(result['artifacts']):
                bbox = artifact.get('bbox_2d', [])
                explanation = artifact.get('explanation', f'Pred {i+1}')
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=3, edgecolor='red', 
                                           facecolor='none', label=f'Pred {i+1}')
                    ax.add_patch(rect)
                    
                    # Add text label
                    ax.text(x1, y1-10, f"PRED: {explanation[:20]}", 
                           color='red', fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Draw ground truth polygons (green filled areas)
        if 'refs' in sample:
            for i, ref in enumerate(sample['refs']):
                if 'segmentation' in ref and ref['segmentation']:
                    seg = ref['segmentation'][0]
                    if len(seg) >= 6:  # At least 3 points
                        x_coords = seg[::2]
                        y_coords = seg[1::2]
                        
                        # Create polygon patch
                        polygon = patches.Polygon(list(zip(x_coords, y_coords)), 
                                                closed=True, facecolor='green', 
                                                alpha=0.3, edgecolor='green', 
                                                linewidth=2, label=f'GT {i+1}')
                        ax.add_patch(polygon)
                        
                        # Add text label
                        explanation = ref.get('explanation', f'GT {i+1}')
                        if x_coords and y_coords:
                            ax.text(x_coords[0], y_coords[0]-15, f"GT: {explanation[:20]}", 
                                   color='green', fontsize=10, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add legend and formatting
        ax.axis('off')
        
        # Create custom legend
        red_patch = patches.Patch(color='red', label='Predictions (Bboxes)')
        green_patch = patches.Patch(color='green', alpha=0.3, label='Ground Truth (Polygons)')
        ax.legend(handles=[red_patch, green_patch], loc='upper right')
        
        # Add info text
        info_text = f"Predicted: {num_artifacts} artifacts | GT: {len(sample.get('refs', []))} artifacts"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / f"synthscars_sample_{sample_idx:04d}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def visualize_richhf(self, image: Image.Image, result: Dict, sample: Dict, 
                         sample_idx: int, save_path: str = None) -> str:
        """
        Visualize RichHF-18K dataset results with predictions and artifact heatmaps.
        
        Args:
            image: PIL Image
            result: Model prediction results
            sample: Ground truth sample data from TFRecord
            sample_idx: Sample index for naming
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization
        """
        artifact_map = sample['artifact_map']

        # Use matplotlib for better heatmap visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Top left: Original image with predictions
        ax.imshow(image)
        ax.axis('off')
        
        # Draw predicted bounding boxes (red)
        num_artifacts = result.get('number_of_artifacts', 0)
        if num_artifacts > 0 and 'artifacts' in result:
            for i, artifact in enumerate(result['artifacts']):
                bbox = artifact['bbox_2d']
                explanation = artifact.get('explanation', f'Pred {i+1}')
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=3, edgecolor='red', 
                                           facecolor='none', alpha=0.8)
                    ax.add_patch(rect)

        ax.imshow(artifact_map, cmap='hot', alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / f"richhf_sample_{sample_idx:04d}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def save_summary_report(self, results: List[Dict], dataset_type: str, 
                           output_path: str = None) -> str:
        """
        Save a summary report of visualization results.
        
        Args:
            results: List of evaluation results
            dataset_type: Type of dataset
            output_path: Optional custom output path
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = self.output_dir / f"{dataset_type}_visualization_report.txt"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write(f"Visualization Report: {dataset_type.upper()} Dataset\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples visualized: {len(results)}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Sample {i+1:04d}:\n")
                f.write(f"  Image: {result.get('image_path', 'Unknown')}\n")
                f.write(f"  Visualization: {result.get('visualization_path', 'Not saved')}\n")
                f.write(f"  Binary Success: {result.get('binary_success', False)}\n")
                f.write(f"  IoU Score: {result.get('iou', 0.0):.3f}\n")
                f.write(f"  ROUGE-L: {result.get('rouge_l', 0.0):.3f}\n")
                f.write(f"  CSS Score: {result.get('css', 0.0):.3f}\n")
                f.write("\n")
        
        return str(output_path)


class Evaluation:
    """
    Provides utilities for evaluating artifact detection performance.
    
    This class computes various metrics including IoU, ROUGE-L scores,
    and cosine similarity scores for artifact detection evaluation.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        if SentenceTransformer is not None:
            self.css_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        else:
            self.css_model = None
            
        if rouge_scorer is not None:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None

    @staticmethod
    def _compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            boxA: Bounding box in format [x1, y1, x2, y2]
            boxB: Bounding box in format [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        # Determine intersection coordinates
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute intersection area
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        if interArea == 0:
            return 0.0

        # Compute union area
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        
        return interArea / float(unionArea) if unionArea > 0 else 0.0

    @staticmethod
    def _compute_iou_polygon(seg: List[float], bbox: List[float]) -> float:
        """
        Compute the Intersection over Union (IoU) of a polygonal segmentation and a bbox.
        
        Args:
            seg: Polygon coordinates as flat list [x1, y1, x2, y2, ..., xn, yn]
            bbox: Bounding box in format [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        try:
            # Convert flat coordinates to point pairs
            poly_points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            
            # Create Shapely polygons
            seg_poly = Polygon(poly_points)
            bbox_poly = box(*bbox)
            
            # Compute intersection and union
            intersection = seg_poly.intersection(bbox_poly).area
            union = seg_poly.union(bbox_poly).area
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            # Fallback to 0 if polygon operations fail
            return 0.0

    @staticmethod
    def _compute_iou_heatmap(artifact_map: np.ndarray, bbox: List[float], 
                           image_width: int, image_height: int, 
                           threshold: float = 0.5) -> float:
        """
        Compute the Intersection over Union (IoU) between a heatmap and a bounding box.
        
        Args:
            artifact_map: 2D heatmap array of shape (512, 512) with values in [0, 1]
            bbox: Bounding box in format [x1, y1, x2, y2] in image coordinates
            image_width: Original image width
            image_height: Original image height  
            threshold: Threshold to binarize the heatmap (default: 0.5)
            
        Returns:
            IoU score between 0 and 1
        """
        try:
            # Ensure heatmap is 2D
            if len(artifact_map.shape) == 3:
                artifact_map = artifact_map[:, :, 0]
            
            heatmap_h, heatmap_w = artifact_map.shape
            
            # Normalize bbox coordinates to heatmap coordinates
            x1, y1, x2, y2 = bbox
            heatmap_x1 = int((x1 / image_width) * heatmap_w)
            heatmap_y1 = int((y1 / image_height) * heatmap_h)  
            heatmap_x2 = int((x2 / image_width) * heatmap_w)
            heatmap_y2 = int((y2 / image_height) * heatmap_h)
            
            # Clamp coordinates to heatmap bounds
            heatmap_x1 = max(0, min(heatmap_x1, heatmap_w - 1))
            heatmap_y1 = max(0, min(heatmap_y1, heatmap_h - 1))
            heatmap_x2 = max(0, min(heatmap_x2, heatmap_w - 1))
            heatmap_y2 = max(0, min(heatmap_y2, heatmap_h - 1))
            
            # Create binary mask from heatmap using threshold
            binary_heatmap = (artifact_map > threshold).astype(np.float32)
            
            # Create binary bbox mask
            bbox_mask = np.zeros_like(binary_heatmap)
            bbox_mask[heatmap_y1:heatmap_y2+1, heatmap_x1:heatmap_x2+1] = 1.0
            
            # Compute intersection and union
            intersection = np.sum(binary_heatmap * bbox_mask)
            union = np.sum(np.maximum(binary_heatmap, bbox_mask))
            
            # Return IoU
            return float(intersection / union) if union > 0 else 0.0
            
        except Exception:
            # Fallback to 0 if computation fails
            return 0.0

    @staticmethod
    def _match_and_mean_iou(
        gt_data: List[Any], 
        est_bboxes: List[List[float]],
        use_polygons: bool = False
    ) -> Tuple[float, Dict[int, int]]:
        """
        Match estimated bboxes to ground truth and compute mean IoU.
        
        For each ground truth annotation, find the best matching estimated bbox
        (highest IoU > 0) and return the mean IoU and mapping.
        
        Args:
            gt_data: Ground truth bboxes or polygon segmentations
            est_bboxes: Estimated bounding boxes
            use_polygons: Whether to use polygon-based IoU computation
            
        Returns:
            Tuple of (mean_iou, mapping_dict)
        """
        if not gt_data or not est_bboxes:
            return 0.0, {}

        mapping = {}  # GT index -> estimated bbox index
        all_ious = []
        
        for gt_idx, gt_item in enumerate(gt_data):
            best_iou = 0.0
            best_est_idx = -1
            
            for est_idx, est_bbox in enumerate(est_bboxes):
                if use_polygons:
                    # gt_item is a polygon segmentation
                    iou = Evaluation._compute_iou_polygon(gt_item, est_bbox)
                else:
                    # gt_item is a bounding box
                    iou = Evaluation._compute_iou(gt_item, est_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_est_idx = est_idx
            
            if best_iou > 0:
                mapping[gt_idx] = best_est_idx
                all_ious.append(best_iou)

        mean_iou = np.mean(all_ious) if all_ious else 0.0
        return mean_iou, mapping

    def _css_score(self, s1: str, s2: str) -> float:
        """
        Compute cosine similarity score between two text strings.
        
        Args:
            s1: First text string
            s2: Second text string
            
        Returns:
            Cosine similarity score
        """
        emb1 = self.css_model.encode(s1, convert_to_tensor=True)
        emb2 = self.css_model.encode(s2, convert_to_tensor=True)
        
        cosine_sim = util.cos_sim(emb1, emb2).cpu().item()
        return cosine_sim

    def get_scores(
        self, 
        ground_data: List[Union[List[float], List[List[float]]]], 
        ground_desc_list: List[str], 
        result_bbox_list: List[List[float]], 
        result_desc_list: List[str],
        use_polygons: bool = False
    ) -> Tuple[float, float, float]:
        """
        Compute evaluation scores for artifact detection results.
        
        Args:
            ground_data: Ground truth bboxes or polygon segmentations
            ground_desc_list: Ground truth descriptions
            result_bbox_list: Predicted bounding boxes
            result_desc_list: Predicted descriptions
            use_polygons: Whether to use polygon-based IoU computation
            
        Returns:
            Tuple of (mean_iou, mean_rouge_l, mean_css)
        """
        rouge_scores = []
        css_scores = []

        iou, mapping = self._match_and_mean_iou(
            ground_data, result_bbox_list, use_polygons
        )

        # Compute text-based scores for matched artifacts
        for gt_idx, est_idx in mapping.items():
            if gt_idx < len(ground_desc_list) and est_idx < len(result_desc_list):
                ref = ground_desc_list[gt_idx]
                hyp = result_desc_list[est_idx]
                
                # ROUGE-L score
                score = self.rouge_scorer.score(ref, hyp)
                rouge_scores.append(score['rougeL'].fmeasure)
                
                # Cosine similarity score
                css_scores.append(self._css_score(ref, hyp))

        # Compute mean scores, penalizing unmatched ground truth artifacts
        num_gt = len(ground_desc_list)
        mean_rouge_l = np.sum(rouge_scores) / num_gt if num_gt > 0 else 0.0
        mean_css = np.sum(css_scores) / num_gt if num_gt > 0 else 0.0

        return iou, mean_rouge_l, mean_css

    def generate_statistics(
        self, 
        dataset_type: str, 
        json_data: Dict, 
        result: Dict
    ) -> Tuple[bool, float, float, float]:
        """
        Generate evaluation statistics for a single image.
        
        Args:
            dataset_type: Type of dataset ('synthscars', 'synartifact', 'loki')
            json_data: Ground truth annotation data
            result: Model prediction results
            
        Returns:
            Tuple of (binary_classification_success, iou, mean_rouge_l, mean_css)
        """
        num_artifacts = result.get('number_of_artifacts', 0)
        
        # Initialize variables for all paths
        stats = {
            'binary_success': False,
            'iou': 0.0,
            'rouge_l': 0.0,
            'css': 0.0,
            'classification': None,  # Will be 'TP', 'FP', 'FN', or 'TN'
            'has_gt_artifacts': False,
            'has_pred_artifacts': False
        }
        
        # Extract prediction data if artifacts were detected
        result_bbox_list = []
        result_desc_list = []
        if num_artifacts > 0 and 'artifacts' in result:
            result_bbox_list = [d.get('bbox_2d', []) for d in result['artifacts'] if 'bbox_2d' in d]
            result_desc_list = [d.get('explanation', '') for d in result['artifacts'] if 'explanation' in d]

        stats['has_pred_artifacts'] = num_artifacts > 0    

        if dataset_type == 'synthscars':
            stats['has_gt_artifacts'] = True    # No negative samples

            # SynthScars uses polygon segmentation
            ground_seg_list = [d['segmentation'][0] for d in json_data['refs'] if 'segmentation' in d]
            ground_desc_list = [d.get('explanation', '') for d in json_data['refs']]
            
            if num_artifacts > 0 and result_bbox_list:
                stats['classification'] = 'TP'
                stats['binary_success'] = True
                stats['iou'], stats['rouge_l'], stats['css'] = self.get_scores(
                    ground_seg_list, ground_desc_list, 
                    result_bbox_list, result_desc_list, 
                    use_polygons=True
                )
            else:
                stats['classification'] = 'FN'
                stats['binary_success'] = False

        elif dataset_type == 'synartifact':
            # SynArtifact contains negative samples
            has_gt_artifacts = json_data.get('Artifacts annotation', [])
            stats['has_gt_artifacts'] = bool(has_gt_artifacts)
            
            if not has_gt_artifacts:
                # No artifacts in ground truth
                stats['binary_success'] = (num_artifacts == 0)
                stats['classification'] = 'TN' if stats['binary_success'] else 'FP'
            else:
                # Ground truth has artifacts
                if num_artifacts > 0 and result_bbox_list:
                    stats['binary_success'] = True
                    stats['classification'] = 'TP'
                    # Convert rect_start + rect_end to bbox format
                    ground_bbox_list = []
                    ground_desc_list = []
                    
                    for annotation in has_gt_artifacts:
                        if 'rect_start' in annotation and 'rect_end' in annotation:
                            bbox = annotation['rect_start'] + annotation['rect_end']
                            ground_bbox_list.append(bbox)
                            ground_desc_list.append(annotation.get('artifacts_caption', ''))
                    
                    if ground_bbox_list:
                        stats['iou'], stats['rouge_l'], stats['css'] = self.get_scores(
                            ground_bbox_list, ground_desc_list,
                            result_bbox_list, result_desc_list,
                            use_polygons=False
                        )
                else:
                    # False negative
                    stats['binary_success'] = False
                    stats['classification'] = 'FN'

        elif dataset_type == 'loki':
            stats['has_gt_artifacts'] = True
            # LOKI dataset format
            regional_problems = json_data['problems']['regional']
            ground_bbox_list = []
            ground_desc_list = []
            
            for problem in regional_problems:
                if 'region' in problem:
                    # Convert [x, y, w, h] to [x1, y1, x2, y2]
                    x, y, w, h = problem['region']
                    bbox = [x, y, x + w, y + h]
                    ground_bbox_list.append(bbox)
                    ground_desc_list.append(problem.get('desc', ''))
            
            if num_artifacts > 0 and result_bbox_list and ground_bbox_list:
                stats['binary_success'] = True
                stats['classification'] = 'TP'
                stats['iou'], stats['rouge_l'], stats['css'] = self.get_scores(
                    ground_bbox_list, ground_desc_list,
                    result_bbox_list, result_desc_list,
                    use_polygons=False
                )
            else:
                stats['binary_success'] = False
                stats['classification'] = 'FN'

        elif dataset_type == 'richhf':
            stats['has_gt_artifacts'] = True
            # RichHF-18K dataset format - use artifact_map for ground truth
            artifact_map = json_data['artifact_map']
            
            if num_artifacts > 0 and result_bbox_list:
                stats['binary_success'] = True
                stats['classification'] = 'TP'
                if artifact_map is not None:
                    # Compute IoU for each predicted bbox and take the maximum
                    iou_scores = []
                    for bbox in result_bbox_list:
                        if len(bbox) == 4:
                            bbox_iou = self._compute_iou_heatmap(
                                artifact_map, bbox, 512, 512, 
                                threshold=0.3  # Lower threshold for more sensitive artifact detection
                            )
                            iou_scores.append(bbox_iou)
                    
                    # Use mean IoU across all predicted bboxes
                    stats['iou'] = float(np.mean(iou_scores)) if iou_scores else 0.0
                
            else:
                # False negative - artifacts exist but none predicted
                stats['binary_success'] = False
                stats['classification'] = 'FN'

        return stats

    @staticmethod
    def compute_f1_metrics(results: List[Dict]) -> Dict:
        """
        Compute F1, precision, recall from detailed evaluation results.
        
        Args:
            results: List of detailed evaluation results
            
        Returns:
            Dictionary with F1 metrics and TP/FP/FN/TN counts
        """
        tp = sum(1 for _, r in results.items() if r.get('classification') == 'TP')
        fp = sum(1 for _, r in results.items() if r.get('classification') == 'FP')
        fn = sum(1 for _, r in results.items() if r.get('classification') == 'FN')
        tn = sum(1 for _, r in results.items() if r.get('classification') == 'TN')
        
        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute accuracy
        accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0
        
        # Compute mean IoU/ROUGE/CSS for TP cases only
        tp_results = [r for _, r in results.items() if r.get('classification') == 'TP']
        if tp_results:
            mean_tp_iou = np.mean([r.get('iou', 0.0) for r in tp_results])
            mean_tp_rouge = np.mean([r.get('rouge_l', 0.0) for r in tp_results])
            mean_tp_css = np.mean([r.get('css', 0.0) for r in tp_results])
        else:
            mean_tp_iou = 0.0
            mean_tp_rouge = 0.0
            mean_tp_css = 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'mean_tp_iou': mean_tp_iou,
            'mean_tp_rouge': mean_tp_rouge,
            'mean_tp_css': mean_tp_css
        }