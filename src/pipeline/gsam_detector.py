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
    
    def _apply_nms_filtering(self, detections, phrases: List[str]) -> Tuple[any, List[str]]:
        """Apply Non-Maximum Suppression to filter overlapping detections."""
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        
        nms_indices = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.nms_threshold
        ).numpy().tolist()

        # Filter detections and phrases
        detections.xyxy = detections.xyxy[nms_indices]
        detections.confidence = detections.confidence[nms_indices]
        detections.class_id = detections.class_id[nms_indices]
        filtered_phrases = [phrases[i] for i in nms_indices]
        
        print(f"After NMS: {len(detections.xyxy)} boxes")
        return detections, filtered_phrases
    
    def _separate_entity_subentity_detections(self, detections, num_entities: int) -> Tuple[List[int], List[int]]:
        """Separate detection indices into entity and subentity categories."""
        entity_indices = []
        subentity_indices = []
        
        for i, class_id in enumerate(detections.class_id):
            if class_id < num_entities:
                entity_indices.append(i)
            else:
                subentity_indices.append(i)
        
        print(f"Found {len(entity_indices)} entity detections and {len(subentity_indices)} subentity detections")
        return entity_indices, subentity_indices
    
    def _calculate_containment_ratio(self, subentity_mask: torch.Tensor, entity_mask: torch.Tensor) -> float:
        """Calculate containment ratio: how much of the subentity is contained within the entity."""
        intersection = torch.sum(subentity_mask & entity_mask).item()
        subentity_area = torch.sum(subentity_mask).item()
        return intersection / subentity_area if subentity_area > 0 else 0.0
    
    def _is_valid_area_ratio(self, mask: np.ndarray, min_ratio: float, max_ratio: float) -> bool:
        """Check if mask area ratio is within valid range."""
        mask_area = np.sum(mask > 0)
        image_area = self.current_image_size[0] * self.current_image_size[1]
        area_ratio = mask_area / image_area
        return min_ratio <= area_ratio <= max_ratio
    
    def _map_subentities_to_entities(self, detections, entity_indices: List[int], 
                                    subentity_indices: List[int], entities: List[str], 
                                    vocab: List[str], entity_mapping: Dict[str, List[str]],
                                    min_area_ratio: float, max_area_ratio: float,
                                    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD) -> Tuple[List[int], List[str], List[str]]:
        """Map subentities to entities using containment ratio and filter by area ratio."""
        
        # Prepare entity masks for containment calculation
        entity_masks = []
        for entity_idx in entity_indices:
            entity_mask = torch.from_numpy(detections.mask[entity_idx])
            entity_class = detections.class_id[entity_idx]
            entity_masks.append((entity_mask, entity_class, entity_idx))
        
        filtered_indices = []
        filtered_subentity_names = []
        filtered_entity_names = []
        
        for subentity_idx in subentity_indices:
            subentity_mask = torch.from_numpy(detections.mask[subentity_idx])
            subentity_name = vocab[detections.class_id[subentity_idx]]
            
            # Skip if subentity not in entity_mapping
            if subentity_name not in entity_mapping:
                print(f"Subentity '{subentity_name}' not found in entity mapping")
                continue
            
            best_containment_ratio = 0.0
            best_entity_class = None
            
            # Find best matching entity
            for entity_mask, entity_class, _ in entity_masks:
                if torch.sum(entity_mask) == 0:  # Skip empty masks
                    continue
                    
                entity_name = entities[entity_class]
                
                # Only consider entities that can contain this subentity
                if entity_name not in entity_mapping[subentity_name]:
                    continue
                
                # Skip if entity is smaller than subentity (illogical)
                if torch.sum(entity_mask) < torch.sum(subentity_mask):
                    continue
                
                # Calculate containment ratio (intersection / subentity_area)
                containment_ratio = self._calculate_containment_ratio(subentity_mask, entity_mask)
                
                if containment_ratio > best_containment_ratio:
                    best_containment_ratio = containment_ratio
                    best_entity_class = entity_class
            
            # Filter by containment threshold and area ratio
            if best_containment_ratio >= containment_threshold and best_entity_class is not None:
                entity_name = entities[best_entity_class]
                
                # Apply area ratio filtering
                if self._is_valid_area_ratio(detections.mask[subentity_idx], min_area_ratio, max_area_ratio):
                    filtered_indices.append(subentity_idx)
                    filtered_subentity_names.append(subentity_name)
                    filtered_entity_names.append(entity_name)
                    print(f"Mapped subentity '{subentity_name}' to entity '{entity_name}' with containment ratio {best_containment_ratio:.3f}")
                else:
                    mask_area = np.sum(detections.mask[subentity_idx] > 0)
                    image_area = self.current_image_size[0] * self.current_image_size[1]
                    area_ratio = mask_area / image_area
                    print(f"Discarded subentity '{subentity_name}' - area ratio {area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]")
            else:
                print(f"Discarded subentity '{subentity_name}' - no valid entity mapping (best containment ratio: {best_containment_ratio:.3f})")
        
        return filtered_indices, filtered_subentity_names, filtered_entity_names
    
    def _create_annotated_image(self, image: np.ndarray, detections, filtered_indices: List[int],
                                filtered_subentity_names: List[str], filtered_entity_names: List[str]) -> Image.Image:
        """Create annotated image with filtered detections."""
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Only plot the filtered detections
        filtered_detections = detections[filtered_indices]
        filtered_labels = [
            f"{entity} ({subentity}) {confidence:0.2f}" 
            for subentity, entity, confidence in zip(
                filtered_subentity_names, 
                filtered_entity_names, 
                detections.confidence[filtered_indices]
            )
        ]
        
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=filtered_detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=filtered_detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=filtered_detections, labels=filtered_labels)
        
        return Image.fromarray(annotated_image)
    
    def _format_predictions(self, detections, filtered_indices: List[int], 
                            filtered_subentity_names: List[str], filtered_entity_names: List[str],
                            entity_indices: List[int], entities: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Format detections into structured prediction dictionaries."""
        
        # Create subentity predictions
        predictions = []
        for i, detection_idx in enumerate(filtered_indices):
            pred_instance = {
                'pred_box': torch.from_numpy(detections.xyxy[detection_idx]).float(),
                'pred_class': torch.tensor(detections.class_id[detection_idx]).long(),
                'score': torch.tensor(detections.confidence[detection_idx]).float(),
                'pred_mask': torch.from_numpy(detections.mask[detection_idx]).bool(),
                'subentity': filtered_subentity_names[i],
                'entity': filtered_entity_names[i],
            }
            predictions.append(pred_instance)
        
        # Create entity predictions
        entity_predictions = []
        for entity_idx in entity_indices:
            entity_pred_instance = {
                'pred_box': torch.from_numpy(detections.xyxy[entity_idx]).float(),
                'pred_class': torch.tensor(detections.class_id[entity_idx]).long(),
                'score': torch.tensor(detections.confidence[entity_idx]).float(),
                'pred_mask': torch.from_numpy(detections.mask[entity_idx]).bool(),
                'entity': entities[detections.class_id[entity_idx]],
            }
            entity_predictions.append(entity_pred_instance)
            
        return predictions, entity_predictions
    
    def detect_parts(self, image: np.ndarray, entities: List[str], subentities: List[str], entity_mapping: Dict[str, List[str]], 
                    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO, max_area_ratio: float = DEFAULT_MAX_AREA_RATIO) -> Tuple[List[Dict], List[Dict], any]:
        """
        Run part detection on image
        
        Args:
            image: Input image as numpy array (RGB format)
            entities: List of entity names to detect
            subentities: List of subentity names to detect  
            entity_mapping: Dictionary mapping subentities to their possible parent entities
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
        
        Returns:
            Tuple of (predictions, entity_predictions, visualized_output):
            - predictions: List of dictionaries, each containing subentity detection with keys:
                'pred_box', 'pred_class', 'score', 'pred_mask', 'subentity', 'entity'
            - entity_predictions: List of dictionaries, each containing entity detection with keys:
                'pred_box', 'pred_class', 'score', 'pred_mask', 'entity'
            - visualized_output: PIL Image with annotations
        """
        # Input validation
        if not entities or not subentities:
            raise ValueError("Both entities and subentities lists must be non-empty")
        if not entity_mapping:
            raise ValueError("Entity mapping dictionary must be provided")
            
        # Store current image size for area calculations
        self.current_image_size = image.shape[:2]
        vocab = entities + subentities

        # Grounding DINO inference
        # Get grounding output
        detections, phrases = self.grounding_model.predict_with_caption(
            image=image,
            caption=", ".join([*entities, *subentities]),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Generate class_id from phrases since predict_with_caption doesn't include it
        detections.class_id = Model.phrases2classes(phrases=phrases, classes=vocab)

        # Apply NMS filtering
        detections, phrases = self._apply_nms_filtering(detections, phrases)

        # Segment detected boxes
        detections.mask = self.segment(image=image, xyxy=detections.xyxy)

        # Separate entity and subentity detections
        entity_indices, subentity_indices = self._separate_entity_subentity_detections(detections, len(entities))
        
        # Map subentities to entities and apply filtering
        filtered_indices, filtered_subentity_names, filtered_entity_names = self._map_subentities_to_entities(
            detections, entity_indices, subentity_indices, entities, vocab, entity_mapping,
            min_area_ratio, max_area_ratio
        )
        
        print(f"After entity mapping and area ratio filtering: {len(filtered_indices)} detections remain")
        
        if len(filtered_indices) == 0:
            raise ValueError("No subentities mapped to entities, returning empty predictions")

        # Create annotated image
        annotated_image = self._create_annotated_image(
            image, detections, filtered_indices, filtered_subentity_names, filtered_entity_names
        )

        # Format predictions
        predictions, entity_predictions = self._format_predictions(
            detections, filtered_indices, filtered_subentity_names, filtered_entity_names,
            entity_indices, entities
        )
        
        print(f"Returning {len(predictions)} mapped subentity detections and {len(entity_predictions)} entity detections")
        return predictions, entity_predictions, annotated_image
    
    def cleanup(self):
        """Clean up model resources"""
        self.grounding_model = None
        self.sam_predictor = None
        self.current_vocabulary = []