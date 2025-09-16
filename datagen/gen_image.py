import torch
import argparse
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from diffusers import FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline
from datasets import load_dataset
import os
import pandas as pd
import random
import csv
from google import genai
from google.genai import types
from google.oauth2 import service_account
import base64
import os
from PIL import Image
from io import BytesIO

class T2I_Model:
    def __init__(self, model: str, device: str = "cuda"):
        self.model_type = model
        self.device = device

        self.aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        
        if model == 'schnell':
            self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload() 
        elif model == 'dev':
            self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        elif model == 'sd3':
            self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
            self.pipe = self.pipe.to("cuda")
        elif model == 'qwen':
            self.pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
            self.pipe = self.pipe.to(device)
        elif model == 'nanobanana':
            service_account_path = "../eval/key/gemini_gcp.json"
            project_id = "gamebench-456108"
            location = "global"

            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path, scopes=scopes
            )
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials,
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
            
        # self.pipe_initialized = self.pipe is not None



    def generate(self, prompt: str, output_path: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate image from prompt and save to output path"""
        # if not self.pipe_initialized:
        #     raise ValueError(f"Model {self.model_type} not properly initialized")
            
        # Set up generator with seed if provided
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.manual_seed(random.randint(0, 2**32 - 1))
        
        # Generate based on model type
        if self.model_type == 'schnell' and self.pipe is not None:
            image = self.pipe(
                prompt,
                guidance_scale=5.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=generator
            ).images[0]
        elif self.model_type == 'dev' and self.pipe is not None:
            image = self.pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=5.0,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator
            ).images[0]
        elif self.model_type == 'sd3' and self.pipe is not None:
            image = self.pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=5.0,
                generator=generator
            ).images[0]
        elif self.model_type == 'qwen':
            positive_magic = "Ultra HD, 4K, cinematic composition"
            negative_prompt = ""

            width, height = self.aspect_ratios["16:9"]

            image = self.pipe(
                prompt=prompt + positive_magic,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=50,
                true_cfg_scale=5.0,
                generator=torch.Generator(device="cuda").manual_seed(42)
            ).images[0]
        elif self.model_type == 'nanobanana':
            model = "gemini-2.5-flash-image-preview"
            contents = [
              types.Content(
                role="user",
                parts=[
                  types.Part(
                    text=prompt
                  )
                ]
              )
            ]

            generate_content_config = types.GenerateContentConfig(
              temperature = 1,
              top_p = 0.95,
              max_output_tokens = 32768,
              response_modalities = ["TEXT", "IMAGE"],
              safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
              ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
              ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
              ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
              )],
            )

            response = self.client.models.generate_content(
              model = model,
              contents = contents,
              config = generate_content_config,
            )
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Save image
        image.save(output_path)
        
        return {
            'prompt': prompt,
            'model': self.model_type,
            'output_path': str(output_path),
            'seed': generator.initial_seed(),
            'timestamp': datetime.now().isoformat()
        }

class DatasetLoader:
    """Handles loading of different caption datasets"""
    
    @staticmethod
    def load_coco_captions() -> List[str]:
        """Load MSCOCO captions"""
        try:
            # Using the coco dataset from HuggingFace
            dataset_path = "/home/jovyan/image-artifacts/src/datagen/data/coco-captions/pair/train-00000-of-00001.parquet"
            df = pd.read_parquet(dataset_path)
             
            captions = []
            for caption1, _ in df.groupby("caption1"):
                captions.append(caption1)
            return captions
        except Exception as e:
            print(f"Warning: Could not load COCO dataset: {e}")
            # Fallback to some sample captions
            return [
                "A red car parked in front of a building",
                "A cat sitting on a windowsill",
                "Children playing in a park on a sunny day",
                "A delicious pizza with various toppings"
            ]
    
    @staticmethod
    def load_imagereward_captions() -> List[str]:
        """Load ImageReward dataset captions"""
        try:
            # Try to load actual ImageReward dataset
            dataset_path = "/home/jovyan/image-artifacts/src/datagen/data/imagereward/imagereward.json"
            with open(dataset_path, "r") as f:
                data = json.load(f)

            captions = []
            for example in data:
                captions.append(example['text'])
            return captions
        except Exception as e:
            print(f"Warning: Could not load ImageReward dataset: {e}")
            # Fallback to sample prompts
            return [
                "A majestic lion standing on a cliff overlooking the savanna",
                "A futuristic city with flying cars and neon lights",
                "An underwater scene with colorful coral and tropical fish", 
                "A cozy cabin in the woods during winter",
                "A steampunk robot in a Victorian setting"
            ]

    @staticmethod
    def load_parti_captions() -> List[str]:
        """Load Parti-Prompts dataset captions"""
        try:
            dataset = load_dataset("nateraw/parti-prompts", split="train")
            captions = []
            for example in dataset:
                challenge = example.get('Challenge', None)
                if challenge in ['Fine-grained Detail', 'Perspective', 'Quantity', 'Simple Detail', 'Style & Format']:
                    captions.append(example['Prompt'])
            return captions
        except Exception as e:
            print(f"Warning: Could not load Parti-Prompts dataset: {e}")
            return [
                "bond"
            ]

    @staticmethod
    def load_fusecap_captions() -> List[str]:
        """Load FuseCap dataset"""
        try:
            dataset_path = "data/fusecap/CC12_FuseCap.json"
            # dataset_path = "data/fusecap/coco_test_fusecap.json"
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                captions = []
                for example in data['caption']:    # For CC12
                    captions.append(example)
                # for example in data:                # For COCO
                #     captions.extend(example['caption'])
            return captions
        except Exception as e:
            print(f"Warning: Could not load FuseCap dataset: {e}")
            return [
                "A bustling street scene with a clock tower in the background. A parked black car sits on the left side of the street, while a white building and a yellow building are visible on the right. The sky is a mix of white and blue, and there are black doors, metal balconies, and a red sign on the yellow building. A concrete step leads up to the building.",
                "A vase of light and dark purple flowers, including pink and purple blooms, sits on a brown wooden table with a round coaster. The flowers are surrounded by a yellow curtain and a wood and yellow curtain in the background."
            ]

    @staticmethod
    def load_sdu_captions() -> List[str]:
        """Load SDU Captioned Photo dataset"""
        try:
            dataset = load_dataset("vicenteor/sbu_captions", split="train")
            captions = []
            for example in dataset:
                captions.append(example['caption'])
            return captions
        except Exception as e:
            print(f"Warning: Could not load SDU Captioned Photo dataset: {e}")
            return [
                "A wooden chair in the living room"
            ]

def load_dataset_captions(dataset_type: str) -> List[str]:
    """Load captions for the specified dataset type"""
    loader = DatasetLoader()
    
    if dataset_type == 'coco':
        return loader.load_coco_captions()
    elif dataset_type == 'imagereward':
        return loader.load_imagereward_captions()
    elif dataset_type == 'parti':
        return loader.load_parti_captions()
    elif dataset_type == 'conceptual':
        return loader.load_conceptual_captions()
    elif dataset_type == 'fusecap':
        return loader.load_fusecap_captions()
    elif dataset_type == 'sdu':
        return loader.load_sdu_captions()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")


def load_all_datasets_captions() -> List[str]:
    """Load captions from all four datasets, combine and shuffle them"""
    loader = DatasetLoader()
    
    print("📚 Loading all datasets...")
    all_captions = []
    
    # Load from each dataset
    # try:
    #     coco_captions = loader.load_coco_captions()
    #     all_captions.extend(coco_captions)
    #     print(f"✅ Loaded {len(coco_captions)} COCO captions")
    # except Exception as e:
    #     print(f"⚠️ Warning: Could not load COCO captions: {e}")
    # try:
    #     imagereward_captions = loader.load_imagereward_captions()
    #     all_captions.extend(imagereward_captions)
    #     print(f"✅ Loaded {len(imagereward_captions)} ImageReward captions")
    # except Exception as e:
    #     print(f"⚠️ Warning: Could not load ImageReward captions: {e}")
    try:
        parti_captions = loader.load_parti_captions()
        random.shuffle(parti_captions)
        all_captions.extend(parti_captions[:1000])
        print(f"✅ Loaded {len(parti_captions)} Parti-Prompts captions")
    except Exception as e:
        print(f"⚠️ Warning: Could not load Parti-Prompts captions: {e}")
    # try:
    #     conceptual_captions = loader.load_conceptual_captions()
    #     all_captions.extend(conceptual_captions)
    #     print(f"✅ Loaded {len(conceptual_captions)} Conceptual captions")
    # except Exception as e:
    #     print(f"⚠️ Warning: Could not load Conceptual captions: {e}")
    try:
        fusecap_captions = loader.load_fusecap_captions()
        random.shuffle(fusecap_captions)
        all_captions.extend(fusecap_captions[:1000])
        print(f"✅ Loaded {len(fusecap_captions)} FuseCap captions")
    except Exception as e:
        print(f"⚠️ Warning: Could not load FuseCap captions: {e}")
    try:
        sdu_captions = loader.load_sdu_captions()
        random.shuffle(sdu_captions)
        all_captions.extend(sdu_captions[:1000])
        print(f"✅ Loaded {len(sdu_captions)} SDU captions")
    except Exception as e:
        print(f"⚠️ Warning: Could not load SDU captions: {e}")
    
    # Shuffle the combined captions
    random.shuffle(all_captions)
    print(f"🔀 Shuffled {len(all_captions)} total captions from all datasets")
    
    return all_captions


def setup_logging(log_dir: str, dataset_name: str) -> logging.Logger:
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"generation_{dataset_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def generate_images_from_captions(
    model: T2I_Model, 
    captions: List[str], 
    output_dir: Path,
    max_samples: Optional[int] = None,
    logger = None
) -> List[Dict[str, Any]]:
    """Generate images from a list of captions"""
    
    # Select random captions if max_samples is specified
    if max_samples and len(captions) > max_samples:
        selected_captions = random.sample(captions, max_samples)
    else:
        selected_captions = captions
    
    results = []
    
    for i, caption in enumerate(selected_captions):
        # Create unique filename
        filename = f"images/image_{i:04d}_{model.model_type}.png"
        output_path = output_dir / filename
        
        try:
            logger.info(f"Generating with caption: {caption}")
            # Generate image
            result = model.generate(caption, str(output_path))
            result['caption_index'] = i
            results.append(result)
            
            logger.info(f"✅ Generated image {i+1}/{len(selected_captions)}: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to generate image {i+1}: {e}")
            results.append({
                'prompt': caption,
                'model': model.model_type,
                'output_path': str(output_path),
                'error': str(e),
                'caption_index': i,
                'timestamp': datetime.now().isoformat()
            })
    
    return results


def generate_images_with_all_models(
    captions: List[str], 
    output_dir: Path,
    device: str,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Generate images using all four models with even sample distribution"""
    
    # Define all model types
    model_types = ['schnell', 'dev', 'sd3', 'qwen', 'nanobanana']
    
    # Split max_samples evenly across models
    if max_samples:
        samples_per_model = max_samples // 4
        remaining_samples = max_samples % 4
    else:
        samples_per_model = len(captions) // 4
        remaining_samples = len(captions) % 4
    
    results = []
    image_counter = 0
    
    # Select captions for all models
    if max_samples and len(captions) > max_samples:
        selected_captions = random.sample(captions, max_samples)
    else:
        selected_captions = captions
    
    caption_idx = 0
    
    for model_idx, model_type in enumerate(model_types):
        print(f"\n🤖 Initializing {model_type} model...")
        
        try:
            # Initialize model
            model = T2I_Model(model_type, device)
            print(f"✅ {model_type} model initialized successfully")
            
            # Calculate samples for this model (add 1 extra for first models if there's remainder)
            current_samples = samples_per_model + (1 if model_idx < remaining_samples else 0)
            
            # Generate images with this model
            for i in range(current_samples):
                if caption_idx >= len(selected_captions):
                    break
                    
                caption = selected_captions[caption_idx]
                
                # Sequential filename without model name
                filename = f"images/{image_counter:04d}.png"
                output_path = output_dir / filename
                
                try:
                    # Generate image
                    result = model.generate(caption, str(output_path))
                    result['caption_index'] = caption_idx
                    result['image_index'] = image_counter
                    results.append(result)
                    
                    print(f"✅ Generated image {image_counter+1} with {model_type}: {filename}")
                    
                except Exception as e:
                    print(f"❌ Failed to generate image {image_counter+1} with {model_type}: {e}")
                    results.append({
                        'prompt': caption,
                        'model': model_type,
                        'output_path': str(output_path),
                        'error': str(e),
                        'caption_index': caption_idx,
                        'image_index': image_counter,
                        'timestamp': datetime.now().isoformat()
                    })
                
                caption_idx += 1
                image_counter += 1
            
            print(f"✅ Completed {current_samples} images with {model_type}")
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ Failed to initialize {model_type} model: {e}")
            # Still increment counters to maintain sequence
            for i in range(samples_per_model + (1 if model_idx < remaining_samples else 0)):
                if caption_idx >= len(selected_captions):
                    break
                caption = selected_captions[caption_idx]
                filename = f"images/{image_counter:04d}.png"
                output_path = output_dir / filename
                
                results.append({
                    'prompt': caption,
                    'model': model_type,
                    'output_path': str(output_path),
                    'error': f"Model initialization failed: {e}",
                    'caption_index': caption_idx,
                    'image_index': image_counter,
                    'timestamp': datetime.now().isoformat()
                })
                
                caption_idx += 1
                image_counter += 1
    
    return results

def save_annotations(results: List[Dict[str, Any]], output_file: Path):
    """Save generation results as JSON annotations"""
    # Create a clean annotation structure
    annotations = {
        'metadata': {
            'total_images': len(results),
            'successful_generations': len([r for r in results if 'error' not in r]),
            'failed_generations': len([r for r in results if 'error' in r]),
            'created_at': datetime.now().isoformat()
        },
        'images': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def main():
    """Main function for image generation."""
    parser = argparse.ArgumentParser(
        description='Generate images with T2I models over captions'
    )
    parser.add_argument('--model', type=str, choices=['schnell', 'dev', 'sd3', 'qwen', 'nanobanana', 'all'], 
                       default='dev', help='T2I diffusion model to use (default: dev). Use "all" to generate with all models')
    parser.add_argument('--dataset', type=str, 
                       choices=['coco', 'imagereward', 'parti', 'conceptual', 'fusecap', 'sdu', 'all'], 
                       default='all', help='Captions to prompt with (default: all datasets combined)')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device for inference (default: cuda:0)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs (default: logs)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory for results (default: none)')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples to generate (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
                       
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if not args.output_dir:
        model_name = "all_models" if args.model == 'all' else args.model
        if args.dataset == 'all':
            output_path = f"gen_{model_name}_all_datasets"
        else:
            output_path = f"gen_{model_name}_{args.dataset}"
    else:
        output_path = args.output_dir

    # Setup logging
    logger = setup_logging(args.log_dir, args.dataset)
    
    if args.dataset == 'all':
        logger.info(f"🚀 Starting generation with ALL DATASETS combined")
    else:
        logger.info(f"🚀 Starting generation with {args.dataset.upper()} captions")

    if args.model == 'all':
        logger.info(f"🤖 Models: ALL MODELS (schnell, dev, sd3, qwen, nanobanana)")
        logger.info(f"📊 Max samples per model: {args.max_samples // 4} (total: {args.max_samples})")
    else:
        logger.info(f"🤖 Model: {args.model}")
        logger.info(f"📊 Max samples: {args.max_samples}")

    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"🔧 Device: {args.device}")

    try:
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset captions
        if args.dataset == 'all':
            logger.info(f"📚 Loading all datasets...")
            captions = load_all_datasets_captions()
            logger.info(f"✅ Loaded {len(captions)} captions from all datasets")
        else:
            logger.info(f"📚 Loading {args.dataset} dataset...")
            captions = load_dataset_captions(args.dataset)
            logger.info(f"✅ Loaded {len(captions)} captions")
        
        image_dir = Path(output_dir / "images")
        image_dir.mkdir(parents=True, exist_ok=True)

        # Generate images
        logger.info(f"🎨 Starting image generation...")
        if args.model == 'all':
            logger.info("🔄 Using all models with sequential generation...")
            results = generate_images_with_all_models(
                captions, 
                output_dir, 
                args.device,
                args.max_samples
            )
        else:
            # Initialize single model
            logger.info(f"🔥 Initializing {args.model} model...")
            model = T2I_Model(args.model, args.device)
            logger.info("✅ Model initialized successfully")
            
            results = generate_images_from_captions(
                model, 
                captions, 
                output_dir, 
                args.max_samples,
                logger
            )
        
        # Save annotations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = "all_datasets" if args.dataset == 'all' else args.dataset
        model_name = "all_models" if args.model == 'all' else args.model
        annotations_file = output_dir / f"annotations_{dataset_name}_{model_name}_{timestamp}.json"
        
        save_annotations(results, annotations_file)
        logger.info(f"📝 Annotations saved to: {annotations_file}")
        
        # Print summary
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        print(f"\n🎉 Generation completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Images saved to: {output_dir}")
        print(f"📝 Annotations saved to: {annotations_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user.")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"\n❌ Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()