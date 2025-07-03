import sys
import os
import argparse
import multiprocessing as mp
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import openai

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

# VLPart imports (assumes VLPart is installed)
try:
    sys.path.append('/home/jovyan/VLPart')
    sys.path.append('/home/jovyan/VLPart/demo')
    from vlpart.config import add_vlpart_config
    from predictor import VisualizationDemo
    from demo import setup_cfg
except ImportError:
    print("Warning: VLPart not found. Please install VLPart and update paths.")

from .prompts import get_entity_subparts, get_entity_subparts_by_type


def find_vlpart_path():
    """Find VLPart installation path"""
    possible_paths = [
        '/home/jovyan/VLPart',
        'VLPart',
        '../VLPart',
        '../../VLPart'
    ]
    
    for path in possible_paths:
        config_path = os.path.join(path, 'configs/joint/swinbase_cascade_lvis_paco.yaml')
        if os.path.exists(config_path):
            return os.path.abspath(path)  # Return absolute path
    
    raise FileNotFoundError(
        f"VLPart installation not found. Tried: {possible_paths}. "
        "Please update the vlpart_config_file and vlpart_model_weights paths manually."
    )


def check_vlpart_setup(vlpart_path: str) -> List[str]:
    """
    Check if VLPart is properly set up with required files
    
    Args:
        vlpart_path: Path to VLPart installation
        
    Returns:
        List of missing files/directories
    """
    required_files = [
        'configs/joint/swinbase_cascade_lvis_paco.yaml',
        'datasets/metadata/lvis_v1_clip_RN50_a+cname.npy',
        'datasets/metadata',
        'models'  # Directory for model weights
    ]
    
    missing = []
    for file_path in required_files:
        full_path = os.path.join(vlpart_path, file_path)
        if not os.path.exists(full_path):
            missing.append(file_path)
    
    return missing


class VLPartDetector:
    """Handler for VLPart part detection model"""
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 model_weights: Optional[str] = None,
                 confidence_threshold: float = 0.3,
                 openai_client: Optional[openai.OpenAI] = None):
        """
        Initialize VLPart detector
        
        Args:
            config_file: Path to VLPart config file (auto-detected if None)
            model_weights: Path to model weights (auto-detected if None)
            confidence_threshold: Confidence threshold for detections
            openai_client: OpenAI client for vocabulary generation
        """
        # Auto-detect VLPart paths if not provided
        if config_file is None or model_weights is None:
            self.vlpart_path = find_vlpart_path()
            
            # Check VLPart setup
            missing_files = check_vlpart_setup(self.vlpart_path)
            if missing_files:
                raise FileNotFoundError(
                    f"VLPart setup incomplete. Missing files/directories:\n" +
                    "\n".join(f"  - {self.vlpart_path}/{f}" for f in missing_files) +
                    "\n\nPlease ensure VLPart is properly installed with all required files."
                )
            
            if config_file is None:
                config_file = os.path.join(self.vlpart_path, 'configs/joint/swinbase_cascade_lvis_paco.yaml')
            if model_weights is None:
                model_weights = os.path.join(self.vlpart_path, 'models/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth')
        else:
            # If paths are provided, extract VLPart path from config file
            self.vlpart_path = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
        
        self.config_file = config_file
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.openai_client = openai_client
        self.demo = None
        
        # Verify paths exist
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"VLPart config file not found: {self.config_file}")
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
    
    def setup_model(self, custom_vocabulary: List[str]):
        """
        Setup VLPart model with custom vocabulary
        
        Args:
            custom_vocabulary: List of part names to detect
        """
        # Store original working directory
        original_cwd = os.getcwd()
        
        try:
            # Change to VLPart directory so relative paths in config work
            os.chdir(self.vlpart_path)
            print(f"Changed working directory to: {self.vlpart_path}")
            
            # Setup arguments
            parser = argparse.ArgumentParser()
            args = parser.parse_args(args=[])
            args.config_file = self.config_file
            args.output = 'output_image'
            args.vocabulary = 'custom'
            args.custom_vocabulary = ', '.join(custom_vocabulary)
            args.confidence_threshold = self.confidence_threshold
            args.opts = ['MODEL.WEIGHTS', self.model_weights, 'VIS.BOX', 'True']
            
            # Setup configuration (now with correct working directory for relative paths)
            cfg = setup_cfg(args)
            
            # Remove unused metadata if it exists
            if "__unused" in MetadataCatalog.keys():
                MetadataCatalog.remove("__unused")
            
            try:
                # Initialize demo
                if self.demo is None:
                    self.demo = VisualizationDemo(cfg, args)
                else:
                    self.demo.set_custom_vocab(', '.join(custom_vocabulary))
                    
                print("VLPart model setup successful!")
            except Exception as e:
                error_msg = str(e)
                if 'datasets/metadata' in error_msg or 'lvis_v1_clip_RN50_a+cname.npy' in error_msg:
                    raise FileNotFoundError(
                        f"VLPart metadata files not found: {error_msg}\n\n"
                        "SOLUTION:\n"
                        "VLPart requires specific metadata files. To fix this:\n\n"
                        "1. Go to your VLPart directory:\n"
                        f"   cd {self.vlpart_path}\n\n"
                        "2. Create the metadata directory:\n"
                        "   mkdir -p datasets/metadata\n\n"
                        "3. Download required files to datasets/metadata/:\n"
                        "   - lvis_v1_clip_RN50_a+cname.npy\n"
                        "   - lvis_v1_train_cat_info.json\n\n"
                        "4. Check VLPart documentation for download links\n"
                        "5. The pipeline will automatically handle path resolution"
                    )
                else:
                    raise Exception(f"VLPart setup error: {error_msg}")
            
            return args
            
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"Restored working directory to: {original_cwd}")
    
    def generate_vocabulary_from_categories(self, categories: List[str]) -> List[str]:
        """
        Generate part vocabulary from category names using OpenAI
        
        Args:
            categories: List of category names (e.g., ['person', 'car'])
            
        Returns:
            List of part names (e.g., ['person head', 'person arm', 'car wheel'])
        """
        if not self.openai_client:
            raise ValueError("OpenAI client required for vocabulary generation")
        
        vocab = []
        for category in categories:
            subparts_result = get_entity_subparts(self.openai_client, category)
            if subparts_result and 'subparts' in subparts_result:
                entity = subparts_result['entity']
                vocab.extend([f"{entity} {part}" for part in subparts_result['subparts']])
        
        return vocab
    
    def generate_subpart_vocab(self, img_array: Optional[np.ndarray] = None) -> List[str]:
        """
        Generate subpart vocabulary for the current or specified image
        
        Args:
            img_array: Image array (numpy array)
        
        Returns:
            List of subpart vocabulary
        """
        if not self.openai_client:
            raise ValueError("OpenAI client required for subpart vocabulary generation")
        
        subparts_result = get_entity_subparts_by_type(self.openai_client, img_array, 'addition')
        entity = subparts_result['entity']
        subpart_vocab = [f"{entity} {part}" for part in subparts_result['subparts']]

        print(f"Generated subpart vocabulary with {len(subpart_vocab)} parts: {subpart_vocab[:5]}..." if len(subpart_vocab) > 5 else f"Generated subpart vocabulary: {subpart_vocab}")

        return subpart_vocab
    
    def sample_target_part(self, 
                          predictions: Dict,
                          vocab: List[str],
                          min_area_ratio: float = 0.05,
                          max_area_ratio: float = 0.8) -> Tuple[any, int, str]:
        """
        Sample a target part for artifact injection
        
        Args:
            predictions: VLPart predictions
            vocab: Vocabulary list
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
            
        Returns:
            Tuple of (sampled_instance, original_index, class_name)
        """
        from .instance_processor import InstanceProcessor
        
        # Sample instance by score with size filtering
        sampled_instance, sampled_idx = InstanceProcessor.sample_instance_by_score(
            predictions, min_area_ratio, max_area_ratio
        )
        
        if sampled_instance is None:
            return None, None, None
        
        class_name = vocab[sampled_instance.pred_classes.item()]
        
        print(f"Sampled part '{class_name}' with score {sampled_instance.scores.item():.3f}")
        
        return sampled_instance, sampled_idx, class_name
    
    def detect_parts(self, image: np.ndarray) -> Tuple[Dict, any]:
        """
        Run part detection on image
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            Tuple of (predictions, visualized_output)
        """
        if self.demo is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        # Convert RGB to BGR for VLPart (if needed)
        image_bgr = image[:, :, ::-1] if image.shape[2] == 3 else image
        
        # Run detection
        predictions, visualized_output = self.demo.run_on_image(image_bgr)
        
        return predictions, visualized_output
    
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
        self.demo = None
        if "__unused" in MetadataCatalog.keys():
            MetadataCatalog.remove("__unused")
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if VLPart is properly set up and available
        
        Returns:
            True if VLPart can be used, False otherwise
        """
        try:
            vlpart_path = find_vlpart_path()
            missing_files = check_vlpart_setup(vlpart_path)
            return len(missing_files) == 0
        except (FileNotFoundError, ImportError):
            return False
    
    @staticmethod
    def get_setup_status() -> Dict[str, any]:
        """
        Get detailed VLPart setup status
        
        Returns:
            Dictionary with setup information
        """
        try:
            vlpart_path = find_vlpart_path()
            missing_files = check_vlpart_setup(vlpart_path)
            
            return {
                'available': len(missing_files) == 0,
                'vlpart_path': vlpart_path,
                'missing_files': missing_files,
                'error': None
            }
        except Exception as e:
            return {
                'available': False,
                'vlpart_path': None,
                'missing_files': [],
                'error': str(e)
            } 