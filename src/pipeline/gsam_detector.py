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

from pipeline.prompts import get_entity_subparts, get_entity_subparts_by_type

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
    
    
    def generate_subpart_vocab(self, img_array: np.ndarray, artifact_type: str) -> List[str]:
        """
        Generate subpart vocabulary for the current or specified image
        
        Args:
            img_array: Image array (numpy array)
        
        Returns:
            List of subpart vocabulary
        """
        if self.openai_client is None:
            raise ValueError("OpenAI client required for subpart vocabulary generation")
        
        subparts_result = get_entity_subparts_by_type(self.openai_client, img_array, artifact_type=artifact_type)
        entity = subparts_result['entity']
        subpart_vocab = [entity]
        subpart_vocab.extend(subparts_result['subparts'])
        print(f"Generated subpart vocabulary with {len(subpart_vocab)} parts: {subpart_vocab[:5]}..." if len(subpart_vocab) > 5 else f"Generated subpart vocabulary: {subpart_vocab}")

        return subpart_vocab
    
    def _get_grounding_output(self, image_tensor: torch.Tensor, caption: str) -> Tuple[torch.Tensor, List[str]]:
        """Get grounding output from GroundingDINO"""
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        # import ipdb; ipdb.set_trace(context=30)
        self.grounding_model = self.grounding_model.model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(image_tensor[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        
        # Get phrases
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        
        return boxes_filt, pred_phrases
    
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
                     min_area_ratio: float = 0.001, max_area_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict], any]:
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
        
        # Create entity masks for each entity class
        # import ipdb;ipdb.set_trace()
        entity_masks = {}
        for entity_idx in range(len(entities)):
            entity_mask = torch.zeros_like(torch.from_numpy(detections.mask[0]), dtype=torch.bool)
            for i in entity_indices:
                if detections.class_id[i] == entity_idx:
                    entity_mask = entity_mask | torch.from_numpy(detections.mask[i])
            entity_masks[entity_idx] = entity_mask
        
        # Map subentities to entities using IoU and apply area ratio filtering
        filtered_indices = []
        filtered_subentity_names = []
        filtered_entity_names = []
        iou_threshold = 0.9  # Minimum IoU to consider a mapping valid
        
        for sub_idx in subentity_indices:
            sub_mask = torch.from_numpy(detections.mask[sub_idx])
            best_iou = 0.0
            best_entity = None
            
            # Calculate IoU with each entity mask
            for entity_idx, entity_mask in entity_masks.items():
                if torch.sum(entity_mask) == 0:  # Skip empty entity masks
                    continue
                    
                # Calculate IoU between subentity mask and entity mask
                sub_mask_area = torch.sum(sub_mask & entity_mask).item()
                intersection = torch.sum(sub_mask & entity_mask).item()
                iou = intersection / sub_mask_area if sub_mask_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_entity = entity_idx
            
            # Only keep subentity if it maps to an entity with sufficient IoU
            if best_iou >= iou_threshold and best_entity is not None:
                subentity_name = vocab[detections.class_id[sub_idx]]
                entity_name = entities[best_entity]
                
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
            print("Warning: No subentities mapped to entities, returning empty predictions")
            # Return empty predictions as list
            predictions = []
            
            # Create entity predictions as list of dictionaries
            entity_predictions = []
            if entity_indices:
                for entity_idx in entity_indices:
                    entity_pred_instance = {
                        'pred_box': torch.from_numpy(detections.xyxy[entity_idx]).float(),
                        'pred_class': torch.tensor(detections.class_id[entity_idx]).long(),
                        'score': torch.tensor(detections.confidence[entity_idx]).float(),
                        'pred_mask': torch.from_numpy(detections.mask[entity_idx]).bool(),
                        'entity_name': entities[detections.class_id[entity_idx]],
                    }
                    entity_predictions.append(entity_pred_instance)
            
            # Create empty annotated image
            annotated_image = Image.fromarray(image.copy())
            return predictions, entity_predictions, annotated_image
        
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
                'subentity_name': filtered_subentity_names[i],
                'mapped_entity_name': filtered_entity_names[i],
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
                    'entity_name': entities[detections.class_id[entity_idx]],
                }
                entity_predictions.append(entity_pred_instance)
        
        print(f"Returning {len(filtered_indices)} mapped subentity detections and {len(entity_predictions)} entity detections")
        return predictions, entity_predictions, annotated_image
    
    def sample_target_part(self, 
                          predictions: Dict,
                          vocab: List[str],
                          min_area_ratio: float = 0.001,
                          max_area_ratio: float = 0.8) -> Tuple[any, int, str]:
        """
        Sample a target part for artifact injection
        
        Args:
            predictions: GSAM predictions
            vocab: Vocabulary list
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
            
        Returns:
            Tuple of (sampled_instance, original_index, class_name)
        """
        # Sample instance by score with size filtering
        if len(predictions['pred_boxes']) == 0:
            return None, None, None
        elif 0 not in predictions['pred_classes']:
            raise ValueError("No entity found in the image")
        else:
            # Get image dimensions for area calculation
            # Create accumulated segmentation map for entity instances (class 0)
            entity_mask = torch.zeros_like(predictions['pred_masks'][0], dtype=torch.bool)
            for i in range(len(predictions['pred_boxes'])):
                if predictions['pred_classes'][i] == 0:
                    entity_mask = entity_mask | predictions['pred_masks'][i]
            
            image_area = self.current_image_size[0] * self.current_image_size[1] if hasattr(self, 'current_image_size') else 1
            # Filter instances by area ratio
            valid_indices = []
            for i in range(len(predictions['pred_boxes'])):
                # Use mask area for more accurate area calculation
                if predictions['pred_classes'][i] != 0:
                    mask = predictions['pred_masks'][i]
                    mask_area = torch.sum(mask).item()
                    area_ratio = mask_area / image_area
                    print(f"Area ratio: {area_ratio}, {min_area_ratio} <= {area_ratio} <= {max_area_ratio}")
                    if min_area_ratio <= area_ratio <= max_area_ratio:
                        # Calculate IoU between entity mask and current part mask
                        intersection = torch.sum(entity_mask & mask).item()
                        mask_area = torch.sum(mask).item()
                        overlap_ratio = intersection / mask_area if mask_area > 0 else 0
                        print(f"Overlap ratio between entity mask and part mask: {overlap_ratio}")
                        if overlap_ratio > 0.9:
                            valid_indices.append(i)
            
            if not valid_indices:
                raise ValueError("No valid target parts found after filtering")
            else:
                # Sample by score (higher score = higher probability)
                scores = [predictions['scores'][i].item() for i in valid_indices]
                # Select instance with highest score
                max_score_idx = np.argmax(scores)
                sampled_idx = valid_indices[max_score_idx]
                
                # Extract the sampled instance
                sampled_instance = {
                    'pred_box': predictions['pred_boxes'][sampled_idx],
                    'pred_class': predictions['pred_classes'][sampled_idx],
                    'score': predictions['scores'][sampled_idx],
                    'pred_mask': predictions['pred_masks'][sampled_idx]
                }
                
        class_name = vocab[sampled_instance['pred_class'].item()]
        
        print(f"Sampled part '{class_name}' with score {sampled_instance['score'].item():.3f}")
        
        return sampled_instance, sampled_idx, class_name
    
    def sample_multiple_target_parts(self, 
                                   predictions: Dict,
                                   min_area_ratio: float = 0.001,
                                   max_area_ratio: float = 0.8,
                                   max_artifacts: int = 3,
                                   min_score_threshold: float = 0.5,
                                   min_spatial_distance: float = 0.3,
                                   entity_subpart_artifacts: Dict[str, Dict[str, List[str]]] = None,
                                   subentity_to_entity: Dict[str, str] = None) -> List[Tuple[any, int, str, str]]:
        """
        Sample multiple target parts for artifact injection with entity-aware filtering and artifact type assignment
        
        Args:
            predictions: GSAM predictions
            vocab: Vocabulary list
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
            max_artifacts: Maximum number of artifacts to sample
            min_score_threshold: Minimum score threshold for inclusion
            min_spatial_distance: Minimum IoU distance between selected artifacts
            entities: Dictionary mapping entity -> artifact_types (NEW)
            subentity_to_entity: Dictionary mapping subentity -> entity (NEW)
            
        Returns:
            List of tuples (sampled_instance, original_index, class_name, artifact_type)
        """
        if len(predictions['pred_boxes']) == 0:
            return []
        
        # Step 1: Filter instances by area ratio and score as before
        valid_instances = []
        for idx, (score, box, mask, class_idx, subentity_name, mapped_entity_name) in enumerate(zip(
            predictions['scores'], predictions['pred_boxes'], 
            predictions['pred_masks'], predictions['pred_classes'], 
            predictions['subentity_names'], predictions['mapped_entity_names']
        )):
            if score < min_score_threshold:
                continue
                
            # Calculate area ratio
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            mask_area = np.sum(mask_np > 0)
            image_area = mask_np.shape[0] * mask_np.shape[1] if hasattr(self, 'current_image_size') else 1
            area_ratio = mask_area / image_area
            
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
                
            valid_instances.append((score, box, mask, class_idx, idx, subentity_name, mapped_entity_name))
        
        if not valid_instances:
            return []
        
        # Step 3: Maintain separate entity masks and filter overlaps within entities
        filtered_instances = []

        valid_instances.sort(key=lambda x: x[0], reverse=True)
        for score, box, mask, class_idx, idx, subentity_name, mapped_entity_name in valid_instances:
            # Sort by score for this entity
            
            overlap_found = False
            for selected_score, selected_box, selected_mask, _, _, _, _ in filtered_instances:
                iou = self._calculate_iou(box, selected_box)
                if iou > (1.0 - min_spatial_distance):
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered_instances.append((score, box, mask, class_idx, idx, subentity_name, mapped_entity_name))
                    
        # Step 4: Sample maximum artifacts across all entities
        if len(filtered_instances) > max_artifacts:
            # Sort by score and take top instances
            filtered_instances.sort(key=lambda x: x[0], reverse=True)
            filtered_instances = filtered_instances[:max_artifacts]
        
        # Step 5: Assign artifact types randomly for each sampled instance
        sampled_with_artifact_types = []
        for score, box, mask, class_idx, idx, subentity_name, mapped_entity_name in filtered_instances:
            
            # Get available artifact types for this entity
            available_artifact_types = entity_subpart_artifacts[mapped_entity_name][subentity_name]
            selected_artifact_type = random.choice(available_artifact_types)

            
            # Create instance object
            instance = {
                'pred_box': box,
                'pred_mask': mask,
                'score': score,
                'pred_class': class_idx,
                'artifact_type': selected_artifact_type,
                'mapped_entity_name': mapped_entity_name,
                'subentity_name': subentity_name
            }
            
            sampled_with_artifact_types.append(instance)
            print(f"Selected part '{subentity_name}' with score {score:.3f} for {selected_artifact_type} artifact")
        
        print(f"Sampled {len(sampled_with_artifact_types)} target parts with entity-aware filtering")
        return sampled_with_artifact_types
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Convert to numpy if needed
        if hasattr(box1, 'cpu'):
            box1 = box1.cpu().numpy()
        if hasattr(box2, 'cpu'):
            box2 = box2.cpu().numpy()
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_detection_info(self, predictions: Dict, vocab: List[str]) -> List[Dict]:
        """
        Extract detection information from predictions
        
        Args:
            predictions: Model predictions
            vocab: Vocabulary list
            
        Returns:
            List of detection dictionaries with bbox, score, class info
        """
        instances = predictions['instances']
        detections = []
        
        for i in range(len(instances)):
            bbox = instances.pred_boxes.tensor[i].cpu().numpy()
            score = instances.scores[i].item()
            class_idx = instances.pred_classes[i].item()
            class_name = vocab[class_idx] if class_idx < len(vocab) else f"class_{class_idx}"
            
            detection = {
                'bbox': bbox,  # [xmin, ymin, xmax, ymax]
                'score': score,
                'class_name': class_name,
                'class_idx': class_idx,
                'index': i
            }
            detections.append(detection)
        
        return detections
    
    def cleanup(self):
        """Clean up model resources"""
        self.grounding_model = None
        self.sam_predictor = None
        self.current_vocabulary = []