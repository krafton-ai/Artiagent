"""
Standalone LEGION response generator that works in the legion1.4.7 environment.
This script has minimal imports and sets up the LEGION model paths correctly.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add LEGION paths to sys.path
LEGION_BASE_DIR = "/home/jhpark/LEGION"
legion_paths = [
    LEGION_BASE_DIR,
    os.path.join(LEGION_BASE_DIR, "src"),
    os.path.join(LEGION_BASE_DIR, "model"),
    os.path.join(LEGION_BASE_DIR, "tools"),
    os.path.join(LEGION_BASE_DIR, "eval"),
]

for path in legion_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        logger.info(f"Added to Python path: {path}")

try:
    # Basic imports
    import cv2
    import bleach  
    import torch
    import re
    from transformers import CLIPImageProcessor, AutoTokenizer
    
    # LEGION-specific imports
    from model.Legion import LegionForCausalLM
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from model.SAM.utils.transforms import ResizeLongestSide
    from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from eval.utils import grounding_image_ecoder_preprocess
    
    LEGION_AVAILABLE = True
    logger.info("✅ LEGION imports successful")
    
except Exception as e:
    logger.error(f"❌ LEGION import failed: {e}")
    logger.error(f"Python path: {sys.path}")
    LEGION_AVAILABLE = False
    sys.exit(1)


class StandaloneLegionGenerator:
    """Standalone LEGION model for pre-generating responses"""
    
    def __init__(self):
        if not LEGION_AVAILABLE:
            raise ImportError("LEGION model dependencies are not available")
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.instruction = 'Please provide a detailed analysis of artifacts in this photo, considering physical artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved segmentation masks for the corresponding parts of the answer.'
        self._load_model()

    def _load_model(self) -> None:
        """Load LEGION model"""
        logger.info("Loading LEGION model...")
        base_dir = "/data2/jhpark/LEGION/exp/Legion/final_model/global_step7030"
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"LEGION model directory not found: {base_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_dir, cache_dir=None, model_max_length=512, 
            padding_side="right", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        
        seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        torch_dtype = torch.bfloat16
        kwargs = {"torch_dtype": torch_dtype}

        self.model = LegionForCausalLM.from_pretrained(
            base_dir, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs
        )
        
        # Update model config
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Initialize vision modules
        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        # Move to GPU
        self.model = self.model.bfloat16().cuda()
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(device=self.device)

        # Initialize processors
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(1024)
        
        self.model.eval()
        logger.info("✅ LEGION model loaded successfully")

    def _legion_inference(self, image_np):
        """Run LEGION inference on image"""
        instructions = bleach.clean(self.instruction)
        instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

        use_mm_start_end = True

        # Prepare prompt
        conv = conversation_lib.conv_templates['llava_v1'].copy()
        conv.messages = []
        begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        prompt = begin_str + instructions
        if use_mm_start_end:
            replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        image_clip = (self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
        image_clip = image_clip.bfloat16()

        # Preprocess image for grounding encoder
        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = (grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
        image = image.bfloat16()

        # Prepare inputs for inference
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        bboxes = None

        # Generate output
        output_ids, pred_masks = self.model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                                max_tokens_new=512, bboxes=bboxes)
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        # Post-processing
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]

        cleaned_str = re.sub(r'<.*?>', '', text_output)
        pattern = re.compile(r'<p>(.*?)<\/p>')
        phrases = pattern.findall(text_output)
        phrases = [p.strip() for p in phrases]

        # Remove [SEG] token
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()
        
        return cleaned_str, pred_masks, phrases

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        """Generate response for single image"""
        try:
            # Convert PIL to numpy
            image_np = np.array(image.convert('RGB'))
            
            result_caption, pred_masks, phrases = self._legion_inference(image_np)

            pred_masks_tensor = pred_masks[0].cpu()
            binary_pred_masks = pred_masks_tensor > 0
            pred_mask = torch.any(binary_pred_masks, dim=0).int()

            return {"heatmap": pred_mask, "explanation": result_caption}
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return {"heatmap": None, "explanation": "", "error": str(e)}


class SimpleDatasetIterator:
    """Simple iterator over dataset samples without importing evaluation modules"""
    
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
                raise FileNotFoundError(f"Eval file not found: {eval_set}")
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
            
        if not self.json_path.exists():
            raise FileNotFoundError(f"Annotations not found: {self.json_path}")
            
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
    parser = argparse.ArgumentParser(description="Standalone LEGION response generator")
    parser.add_argument('--datasets', nargs='+', default=['synthscars', 'synartifact', 'loki', 'richhf'],
                       help='Datasets to process')
    parser.add_argument('--output_dir', default='/data2/jhpark/image-artifacts/eval/legion_responses',
                       help='Output directory for pre-generated responses')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per dataset (None for all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LEGION model
    logger.info("Initializing LEGION model...")
    generator = StandaloneLegionGenerator()
    
    # Process each dataset
    for dataset in args.datasets:
        logger.info(f"🚀 Processing dataset: {dataset}")
        
        try:
            iterator = SimpleDatasetIterator(dataset)
            responses = {}
            
            total_samples = len(iterator)
            if args.max_samples:
                total_samples = min(total_samples, args.max_samples)
                
            logger.info(f"Processing {total_samples} samples from {dataset}")
            
            for i, (json_data, image_path) in enumerate(iterator):
                if args.max_samples and i >= args.max_samples:
                    break
                    
                logger.info(f"Processing {i+1}/{total_samples}: {image_path.name}")
                
                try:
                    # Load image
                    image = Image.open(str(image_path)).convert("RGB")
                    if dataset == 'richhf':
                        image = image.resize((512, 512), Image.LANCZOS)
                    
                    # Generate response
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
                
            logger.info(f"✅ Saved {len(responses)} responses to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            continue
    
    logger.info("🎉 Pre-generation completed!")


if __name__ == "__main__":
    main()
