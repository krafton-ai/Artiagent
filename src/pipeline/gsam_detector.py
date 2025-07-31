import sys
import os
import argparse
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import cv2
import random
from typing import List, Dict, Tuple, Optional
import openai
import torch
from PIL import Image
import json
import torchvision
import supervision as sv

# Add GroundingDINO and SAM to path
sys.path.append(os.path.join(os.getcwd(), 'GroundingDINO'))
sys.path.append(os.path.join(os.getcwd(), 'segment_anything'))
sys.path.append(os.path.join(os.getcwd(), 'pipeline'))

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import Model

# Segment Anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

from pipeline.prompts import get_entity_subparts_by_type

class GSAMDetector:
    """Handler for Grounded SAM part detection model"""
    
    def __init__(self, 
                 grounding_config_file: Optional[str] = None,
                 grounding_checkpoint: Optional[str] = None,
                 sam_version: str = "vit_h",
                 sam_checkpoint: Optional[str] = None,
                 sam_hq_checkpoint: Optional[str] = None,
                 use_sam_hq: bool = False,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.25,
                 nms_threshold: float = 0.5,
                 bert_base_uncased_path: Optional[str] = None,
                 device: str = "cuda",
                 openai_client: Optional[openai.OpenAI] = None):
        """
        Initialize GSAM detector
        
        Args:
            grounding_config_file: Path to GroundingDINO config file
            grounding_checkpoint: Path to GroundingDINO checkpoint
            sam_version: SAM model version (vit_b, vit_l, vit_h)
            sam_checkpoint: Path to SAM checkpoint
            sam_hq_checkpoint: Path to SAM-HQ checkpoint
            use_sam_hq: Whether to use SAM-HQ
            box_threshold: Box threshold for detection
            text_threshold: Text threshold for detection
            nms_threshold: NMS threshold for detection
            bert_base_uncased_path: Path to BERT model
            device: Device to use (cuda/cpu)
            openai_client: OpenAI client for vocabulary generation
        """
        self.gsam_path = os.getcwd()
        
        # Set default paths if not provided
        if grounding_config_file is None:
            grounding_config_file = os.path.join(self.gsam_path, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        if grounding_checkpoint is None:
            grounding_checkpoint = os.path.join(self.gsam_path, "weight/groundingdino_swint_ogc.pth")
        if sam_checkpoint is None and not use_sam_hq:
            sam_checkpoint = os.path.join(self.gsam_path, "weight/sam_vit_h_4b8939.pth")
        if sam_hq_checkpoint is None and use_sam_hq:
            sam_hq_checkpoint = os.path.join(self.gsam_path, "weight/sam_hq_vit_h.pth")
        
        self.grounding_config_file = grounding_config_file
        self.grounding_checkpoint = grounding_checkpoint
        self.sam_version = sam_version
        self.sam_checkpoint = sam_checkpoint
        self.sam_hq_checkpoint = sam_hq_checkpoint
        self.nms_threshold = nms_threshold
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.openai_client = openai_client
        
        # Model components
        self.grounding_model = Model(model_config_path=self.grounding_config_file, model_checkpoint_path=self.grounding_checkpoint)
        if use_sam_hq:
            self.sam_predictor = SamPredictor(sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            self.sam_predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device))
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
    
    
    # Prompting SAM with detected boxes (same as original)
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def detect_parts(self, image: np.ndarray, entities: List[str], subentities: List[str], 
                     min_area_ratio: float = 0.005, max_area_ratio: float = 0.5) -> Tuple[List[Dict], List[Dict], any]:
        """
        Run part detection on image
        
        Args:
            image: Input image as numpy array (RGB format)
            entities: List of entity names
            subentities: List of subentity names
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
        
        Returns:
            Tuple of (predictions, entity_predictions, visualized_output):
            - predictions: List of dictionaries, each containing subentity detection with keys:
              'pred_box', 'pred_class', 'score', 'pred_mask', 'subentity_name', 'mapped_entity_name'
            - entity_predictions: List of dictionaries, each containing entity detection with keys:
              'pred_box', 'pred_class', 'score', 'pred_mask', 'entity_name'
            - visualized_output: PIL Image with annotations
        """
        # Store current image size for area calculations
        self.current_image_size = image.shape[:2]
        
        # Create vocabulary from entities and subentities
        vocab = entities + subentities

        # Get grounding output
        detections, phrases = self.grounding_model.predict_with_caption(
            image=image,
            caption=", ".join(entities) +", " +  ", ".join(subentities),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Generate class_id from phrases since predict_with_caption doesn't include it
        detections.class_id = Model.phrases2classes(phrases=phrases, classes=vocab)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        # Also filter phrases to match the filtered detections
        phrases = [phrases[i] for i in nms_idx]

        detections.mask = self.segment(
            image=image,
            xyxy=detections.xyxy
        )

        # Separate entity and subentity detections
        entity_indices = []
        subentity_indices = []
        
        for i, class_id in enumerate(detections.class_id):
            if class_id < len(entities):  # Entity detection
                entity_indices.append(i)
            else:  # Subentity detection
                subentity_indices.append(i)
        
        print(f"Found {len(entity_indices)} entity detections and {len(subentity_indices)} subentity detections")
        
        # Create individual entity masks with class information
        entity_masks = []  # List of tuples: (mask, class_idx, detection_idx)
        for i in entity_indices:
            entity_mask = torch.from_numpy(detections.mask[i])
            entity_class = detections.class_id[i]
            entity_masks.append((entity_mask, entity_class, i))
        
        # Map subentities to entities using IoU and apply area ratio filtering
        filtered_indices = []
        filtered_subentity_names = []
        filtered_entity_names = []
        iou_threshold = 0.9  # Minimum IoU to consider a mapping valid
        
        for sub_idx in subentity_indices:
            sub_mask = torch.from_numpy(detections.mask[sub_idx])
            best_iou = 0.0
            best_entity_class = None
            
            # Calculate IoU with each individual entity mask
            for entity_mask, entity_class, _ in entity_masks:
                if torch.sum(entity_mask) == 0:  # Skip empty entity masks
                    continue
                    
                # Calculate IoU between subentity mask and entity mask
                sub_mask_area = torch.sum(sub_mask & entity_mask).item()
                intersection = torch.sum(sub_mask & entity_mask).item()
                iou = intersection / sub_mask_area if sub_mask_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_entity_class = entity_class
            
            # Only keep subentity if it maps to an entity with sufficient IoU
            if best_iou >= iou_threshold and best_entity_class is not None:
                subentity_name = vocab[detections.class_id[sub_idx]]
                entity_name = entities[best_entity_class]
                
                # Apply area ratio filtering
                mask = detections.mask[sub_idx]
                mask_area = np.sum(mask > 0)
                image_area = self.current_image_size[0] * self.current_image_size[1]
                area_ratio = mask_area / image_area
                
                if min_area_ratio <= area_ratio <= max_area_ratio:
                    filtered_indices.append(sub_idx)
                    filtered_subentity_names.append(subentity_name)
                    filtered_entity_names.append(entity_name)
                    print(f"Mapped subentity '{subentity_name}' to entity '{entity_name}' with IoU {best_iou:.3f}")
                else:
                    print(f"Discarded subentity '{subentity_name}' - area ratio {area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]")
            else:
                print(f"Discarded subentity '{vocab[detections.class_id[sub_idx]]}' - no valid entity mapping (best IoU: {best_iou:.3f})")
        
        print(f"After entity mapping and area ratio filtering: {len(filtered_indices)} detections remain")
        
        # Use the filtered results from above
        
        if len(filtered_indices) == 0:
            raise ValueError("No subentities mapped to entities, returning empty predictions")
            
        # Filter detections to keep only mapped subentities
        filtered_xyxy = detections.xyxy[filtered_indices]
        filtered_confidence = detections.confidence[filtered_indices]
        filtered_class_id = detections.class_id[filtered_indices]
        filtered_mask = detections.mask[filtered_indices]

        # annotate image with filtered detections
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        labels = [
            f"{vocab[class_id]} {confidence:0.2f}" 
            for class_id, confidence in zip(detections.class_id, detections.confidence)]
        
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        # Convert annotated image to PIL Image
        annotated_image = Image.fromarray(annotated_image)

        # Create predictions structure from filtered detections
        boxes_tensor = torch.from_numpy(filtered_xyxy).float()
        scores_tensor = torch.from_numpy(filtered_confidence).float()
        classes_tensor = torch.from_numpy(filtered_class_id).long()
        masks_tensor = torch.from_numpy(filtered_mask).bool() if filtered_mask is not None else torch.empty(0, 0, 0, dtype=torch.bool)
        
        # Create predictions as list of dictionaries
        predictions = []
        for i in range(len(boxes_tensor)):
            pred_instance = {
                'pred_box': boxes_tensor[i],
                'pred_class': classes_tensor[i],
                'score': scores_tensor[i],
                'pred_mask': masks_tensor[i] if len(masks_tensor) > 0 else torch.empty(0, 0, dtype=torch.bool),
                'subentity': filtered_subentity_names[i],
                'entity': filtered_entity_names[i],
            }
            predictions.append(pred_instance)
        
        # Create entity predictions as list of dictionaries
        entity_predictions = []
        if entity_indices:
            for entity_idx in entity_indices:
                entity_pred_instance = {
                    'pred_box': torch.from_numpy(detections.xyxy[entity_idx]).float(),
                    'pred_class': torch.tensor(detections.class_id[entity_idx]).long(),
                    'score': torch.tensor(detections.confidence[entity_idx]).float(),
                    'pred_mask': torch.from_numpy(detections.mask[entity_idx]).bool(),
                    'entity': entities[detections.class_id[entity_idx]],
                }
                entity_predictions.append(entity_pred_instance)
        
        print(f"Returning {len(filtered_indices)} mapped subentity detections and {len(entity_predictions)} entity detections")
        return predictions, entity_predictions, annotated_image
    
    def cleanup(self):
        """Clean up model resources"""
        self.grounding_model = None
        self.sam_predictor = None
        self.current_vocabulary = []