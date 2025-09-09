import sys
import os
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
import openai
import torch
from PIL import Image
import torchvision
import supervision as sv
from flux.artifacts_util import mask_to_patch_coords

# Add GroundingDINO and SAM to path
sys.path.append(os.path.join(os.getcwd(), 'GroundingDINO'))
sys.path.append(os.path.join(os.getcwd(), 'segment_anything'))
sys.path.append(os.path.join(os.getcwd(), 'pipeline'))

# Grounding DINO
from groundingdino.util.inference import Model

# S
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

class GSAMDetector:
    """Handler for Grounded SAM part detection model"""
    
    # Constants for better code maintainability
    DEFAULT_CONTAINMENT_THRESHOLD = 0.9
    DEFAULT_MIN_AREA_RATIO = 0.005
    DEFAULT_MAX_AREA_RATIO = 0.5
    
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
                    entity_subentity_mapping: Dict[str, List[str]],
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
            caption=", ".join([*entities, *subentities]),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
            # box_threshold=0.3,
            # text_threshold=0.25
        )
        
        # Generate class_id from phrases since predict_with_caption doesn't include it
        detections.class_id = Model.phrases2classes(phrases=phrases, classes=vocab)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.nms_threshold
            # self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        # Also filter phrases to match the filtered detections
        phrases = [phrases[i] for i in nms_idx]

        detections.mask = self.segment(
            image=image,
            xyxy=detections.xyxy,
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
        filtered_entities = []  # List of tuples: (mask, class_idx, detection_idx)
        image_area = self.current_image_size[0] * self.current_image_size[1]
        for i in entity_indices:
            entity_mask = torch.from_numpy(detections.mask[i])
            entity_class = detections.class_id[i]
            entity_name = entities[entity_class]
            area_ratio = torch.sum(entity_mask > 0) / image_area
            if 0.01 <= area_ratio:
                filtered_entities.append((entity_mask, entity_class, entity_name, i))
                print(f"Kept entity '{entity_name}' (class {entity_class}) with area ratio {area_ratio:.4f}")
            else:
                print(f"Discarded entity '{entity_name}' (class {entity_class}) - area ratio {area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]")
        
        # Map subentities to entities using containment ratio and apply area ratio filtering
        filtered_subentities = []
        containment_ratio_threshold = 0.9  # Minimum containment ratio to consider a mapping valid
        
        for sub_idx in subentity_indices:
            sub_mask = torch.from_numpy(detections.mask[sub_idx])
            sub_mask_patch_coords = mask_to_patch_coords(sub_mask.numpy(), patch_size=16)
            # If the mask_patch_coords has length of 1 for either width or height, discard this subentity
            if len(sub_mask_patch_coords) > 0:
                ys, xs = zip(*sub_mask_patch_coords)
                if (max(xs) - min(xs) + 1) == 1 or (max(ys) - min(ys) + 1) == 1:
                    print(f"Discarded subentity '{vocab[detections.class_id[sub_idx]]}' - mask is only 1 patch wide or high")
                    continue

            
            subentity_name = vocab[detections.class_id[sub_idx]]
            best_containment_ratio = 0.0
            best_entity_class = None
            
            # Calculate containment ratio with each individual entity mask
            for entity_mask, entity_class, entity_name, _ in filtered_entities:
                if torch.sum(entity_mask) == 0:  # Skip empty entity masks
                    continue
                
                if entity_name not in entity_subentity_mapping[subentity_name]:
                    continue
                
                # Calculate containment ratio between subentity mask and entity mask
                sub_mask_area = torch.sum(sub_mask & entity_mask).item()
                intersection = torch.sum(sub_mask & entity_mask).item()
                containment_ratio = intersection / sub_mask_area if sub_mask_area > 0 else 0
                
                if containment_ratio > best_containment_ratio:
                    best_containment_ratio = containment_ratio
                    best_entity_class = entity_class
            
            # Only keep subentity if it maps to an entity with sufficient containment ratio
            if best_containment_ratio >= containment_ratio_threshold and best_entity_class is not None:
                entity_name = entities[best_entity_class]
                
                # Apply area ratio filtering
                sub_mask_area = torch.sum(sub_mask > 0)
                area_ratio = sub_mask_area / image_area
                
                if area_ratio <= max_area_ratio and sub_mask_area > 512:
                    filtered_subentities.append((sub_idx, subentity_name, entity_name))
                    print(f"Mapped subentity '{subentity_name}' to entity '{entity_name}' with containment ratio {best_containment_ratio:.3f}")
                else:
                    print(f"Discarded subentity '{subentity_name}' - area ratio {area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]")
            else:
                print(f"Discarded subentity '{subentity_name}' - no valid entity mapping (best containment ratio: {best_containment_ratio:.3f})")
        
        print(f"After entity mapping and area ratio filtering: {len(filtered_subentities)} detections remain")
        
        # Use the filtered results from above
        
        if len(filtered_subentities) == 0 and len(filtered_entities) == 0:
            raise ValueError("No entity or subentity detected")
            
        # Filter detections to keep only mapped subentities
        filtered_indices = [sub_idx for sub_idx, _, _ in filtered_subentities]
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
                'subentity': filtered_subentities[i][1],
                'entity': filtered_subentities[i][2],
            }
            predictions.append(pred_instance)
        
        # Create entity predictions as list of dictionaries
        entity_predictions = []
        if entity_indices:
            for entity_mask, entity_class, entity_name, i in filtered_entities:
                entity_pred_instance = {
                    'pred_box': torch.from_numpy(detections.xyxy[i]).float(),
                    'pred_class': torch.tensor(detections.class_id[i]).long(),
                    'score': torch.tensor(detections.confidence[i]).float(),
                    'pred_mask': torch.from_numpy(detections.mask[i]).bool(),
                    'entity': entity_name,
                }
                entity_predictions.append(entity_pred_instance)
        
        print(f"Returning {len(filtered_indices)} mapped subentity detections and {len(entity_predictions)} entity detections")
        return predictions, entity_predictions, annotated_image
    
    def detect_entities(self, image: np.ndarray, entities: List[str], 
                       min_area_ratio: float = 0.01, max_area_ratio: float = 1.0) -> Tuple[List[Dict], any]:
        """
        Run entity detection on image (entities only, no subentities)
        
        Args:
            image: Input image as numpy array (RGB format)
            entities: List of entity names
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
        
        Returns:
            Tuple of (entity_predictions, visualized_output):
            - entity_predictions: List of dictionaries, each containing entity detection with keys:
                'pred_box', 'pred_class', 'score', 'pred_mask', 'entity_name'
            - visualized_output: PIL Image with annotations
        """
        # Store current image size for area calculations
        self.current_image_size = image.shape[:2]
        
        # Get grounding output
        detections, phrases = self.grounding_model.predict_with_caption(
            image=image,
            caption=", ".join(entities),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Generate class_id from phrases since predict_with_caption doesn't include it
        detections.class_id = Model.phrases2classes(phrases=phrases, classes=entities)
           
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
            xyxy=detections.xyxy,
        )
        
        print(f"Found {len(detections.class_id)} entity detections")
        
        # Filter entities by area ratio
        filtered_entities = []
        image_area = self.current_image_size[0] * self.current_image_size[1]
        
        for i in range(len(detections.class_id)):
            entity_mask = torch.from_numpy(detections.mask[i])
            entity_class = detections.class_id[i]
            entity_name = entities[entity_class]
            area_ratio = torch.sum(entity_mask > 0) / image_area
            
            if min_area_ratio <= area_ratio <= max_area_ratio:
                filtered_entities.append(i)
                print(f"Kept entity '{entity_name}' (class {entity_class}) with area ratio {area_ratio:.4f}")
            else:
                print(f"Discarded entity '{entity_name}' (class {entity_class}) - area ratio {area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]")
        
        if len(filtered_entities) == 0:
            raise ValueError("No entities detected after filtering")
        
        # Filter detections to keep only valid entities
        filtered_xyxy = detections.xyxy[filtered_entities]
        filtered_confidence = detections.confidence[filtered_entities]
        filtered_class_id = detections.class_id[filtered_entities]
        filtered_mask = detections.mask[filtered_entities]

        # Create filtered detections object for annotation
        filtered_detections = sv.Detections(
            xyxy=filtered_xyxy,
            confidence=filtered_confidence,
            class_id=filtered_class_id,
            mask=filtered_mask
        )

        # Annotate image with filtered detections
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        labels = [
            f"{entities[class_id]} {confidence:0.2f}" 
            for class_id, confidence in zip(filtered_class_id, filtered_confidence)]
        
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=filtered_detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=filtered_detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=filtered_detections, labels=labels)
        # Convert annotated image to PIL Image
        annotated_image = Image.fromarray(annotated_image)

        # Create entity predictions as list of dictionaries
        entity_predictions = []
        for i, entity_idx in enumerate(filtered_entities):
            entity_pred_instance = {
                'pred_box': torch.from_numpy(detections.xyxy[entity_idx]).float(),
                'pred_class': torch.tensor(detections.class_id[entity_idx]).long(),
                'score': torch.tensor(detections.confidence[entity_idx]).float(),
                'pred_mask': torch.from_numpy(detections.mask[entity_idx]).bool(),
                'entity': entities[detections.class_id[entity_idx]],
            }
            entity_predictions.append(entity_pred_instance)
        
        print(f"Returning {len(entity_predictions)} entity detections")
        return entity_predictions, annotated_image
    
    def cleanup(self):
        """Clean up model resources"""
        self.grounding_model = None
        self.sam_predictor = None
        self.current_vocabulary = []