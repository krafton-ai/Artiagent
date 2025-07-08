import sys
import os
import argparse
import multiprocessing as mp
import numpy as np
import cv2
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
    
    def detect_parts(self, image: np.ndarray, vocab: List[str]) -> Tuple[Dict, any]:
        """
        Run part detection on image
        
        Args:
            image: Input image as numpy array (RGB format)
            vocab: Vocabulary list
        
        Returns:
            Tuple of (predictions, visualized_output)
        """
        # Store current image size for area calculations
        self.current_image_size = image.shape[:2]

        # Get grounding output
        detections = self.grounding_model.predict_with_classes(
            image=image,
            classes=vocab,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # NMS post process (same as original)
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        detections.mask = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections (same as original)
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{vocab[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        # Convert annotated image to PIL Image
        annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Create predictions structure directly from detections
        # Convert detections to torch tensors for consistency with expected format
        boxes_tensor = torch.from_numpy(detections.xyxy).float()
        scores_tensor = torch.from_numpy(detections.confidence).float()
        classes_tensor = torch.from_numpy(detections.class_id).long()
        masks_tensor = torch.from_numpy(detections.mask).bool() if detections.mask is not None else torch.empty(0, 0, 0, dtype=torch.bool)
        
        predictions = {
            'pred_boxes': boxes_tensor,
            'pred_classes': classes_tensor,
            'scores': scores_tensor,
            'pred_masks': masks_tensor,
        }
        
        return predictions, annotated_image
    
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