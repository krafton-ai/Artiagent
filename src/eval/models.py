"""
Model classes for artifact detection evaluation.

This module contains model wrappers for different VLM/MLLM models
to evaluate their artifact detection capabilities.
"""

import os
import sys
import torch
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import json
import re
import io
import base64
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoConfig, AutoTokenizer, SegformerImageProcessor, SegformerForSemanticSegmentation
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from qwen_vl_utils import process_vision_info
import openai
from openai.types.chat import ChatCompletion
from google.oauth2 import service_account
from google import genai
from google.genai import types


class MoneyManager:
    def __init__(self, model: str = "gpt-3.5-turbo-0613"):
        self.total_cost = 0.0
        self.model = model
        if self.model == "gpt-3.5-turbo-16k-0613":
            self.input_cost = 0.003
            self.output_cost = 0.004
        elif self.model == "gpt-3.5-turbo-1106":
            self.input_cost = 0.001
            self.output_cost = 0.002
        elif self.model == "gpt-3.5-turbo":
            self.input_cost = 0.001
            self.output_cost = 0.002
        elif self.model == "gpt-4-turbo-preview":
            self.input_cost = 0.01
            self.output_cost = 0.03
        elif self.model == "gpt-4-turbo":
            self.input_cost = 0.01
            self.output_cost = 0.03
        elif self.model == "gpt-4-1106-preview":
            self.input_cost = 0.01
            self.output_cost = 0.03
        elif self.model == "gpt-4":
            self.input_cost = 0.03
            self.output_cost = 0.06
        elif self.model == "text-embedding-ada-002":
            self.input_cost = 0.0001
            self.output_cost = 0.0
        elif self.model == "claude-3-opus-20240229":
            self.input_cost = 0.015
            self.output_cost = 0.075
        elif self.model == "claude-opus-4-20250514":
            self.input_cost = 15 / 1000
            self.output_cost = 75 / 1000
        elif self.model == "claude-sonnet-4-20250514":
            self.input_cost = 3 / 1000
            self.output_cost = 15 / 1000
        elif self.model == "gpt-4o":
            self.input_cost = 2.5 / 1000
            self.output_cost = 10 / 1000
        elif self.model == "gpt-4o-mini":
            self.input_cost = 0.15 / 1000
            self.output_cost = 0.6 / 1000
        elif self.model == "gpt-4o-2024-08-06":
            self.input_cost = 2.5 / 1000
            self.output_cost = 10 / 1000
        elif self.model == "gpt-4o-2024-05-13":
            self.input_cost = 5 / 1000
            self.output_cost = 15 / 1000
        elif self.model == "o1-preview":
            self.input_cost = 15 / 1000
            self.output_cost = 60 / 1000
        elif self.model == "o1-preview-2024-09-12":
            self.input_cost = 15 / 1000
            self.output_cost = 60 / 1000
        elif self.model == "o1-2024-12-17":
            self.input_cost = 15 / 1000
            self.output_cost = 60 / 1000
        elif self.model == "o1-mini":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "o1-mini-2024-09-12":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "o3-mini":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "o3":
            self.input_cost = 2 / 1000
            self.output_cost = 8 / 1000
        elif self.model == "o3-mini-2025-01-31":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "gpt-4.1":
            self.input_cost = 2 / 1000
            self.output_cost = 8 / 1000
        elif self.model == "gpt-4.1-mini":
            self.input_cost = 0.4 / 1000
            self.output_cost = 1.6 / 1000
        elif self.model == "gpt-4.1-2025-04-14":
            self.input_cost = 2 / 1000
            self.output_cost = 8 / 1000
        elif self.model == "o4-mini":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "o4-mini-2025-04-16":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "gemini-2.5-flash":
            self.input_cost = 0.3 / 1000
            self.output_cost = 2.5 / 1000
        elif self.model == "gemini-2.5-pro":
            self.input_cost = 1.25 / 1000
            self.output_cost = 10 / 1000
        else:
            print(
                f"MoneyManager: Model {self.model} not found. If you are using a new model, please add the cost to the MoneyManager class."
            )
            self.input_cost = 0.0
            self.output_cost = 0.0

    def __call__(self, response: Union[ChatCompletion, None] = None) -> None:
        if hasattr(response, "usage") and response.usage is None:
            print("No usage in response")
            print(response)
            return

        if self.model == "gemini-2.5-flash" or self.model == "gemini-2.5-pro":
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = (
                response.usage_metadata.candidates_token_count
                + response.usage_metadata.thoughts_token_count
            )

            # Default rates from init
            input_rate_per_token = self.input_cost
            output_rate_per_token = self.output_cost

            # Tiered pricing for gemini-2.5-pro based on prompt size (<= 200k vs > 200k)
            if self.model == "gemini-2.5-pro":
                if input_tokens > 200_000:
                    input_rate_per_token = 2.50 / 1000
                    output_rate_per_token = 15.00 / 1000
                else:
                    input_rate_per_token = 1.25 / 1000
                    output_rate_per_token = 10.00 / 1000

        else:  # OpenAI and Claude
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens 

            if "o1" in self.model or "o3" in self.model or "o4" in self.model:
                output_tokens += (
                    response.usage.completion_tokens_details.accepted_prediction_tokens
                    + response.usage.completion_tokens_details.reasoning_tokens
                    + response.usage.completion_tokens_details.rejected_prediction_tokens
                )

            input_rate_per_token = self.input_cost
            output_rate_per_token = self.output_cost

        input_cost = input_tokens / 1000 * input_rate_per_token
        output_cost = output_tokens / 1000 * output_rate_per_token

        self.total_cost += input_cost + output_cost

    def refresh(self) -> None:
        self.total_cost = 0.0

class QwenEval:
    """
    Wrapper class for Qwen2.5-VL model evaluation.
    
    This class provides a unified interface for running inference
    on images to detect and describe artifacts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Qwen model for evaluation.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_finetuned = config.get('use_finetuned', False)
        
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the Qwen2.5-VL model and processor."""
        if self.use_finetuned:
            model_name = "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft_artifacts_gpt"
            # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.padding_side = "left"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map=self.device)
        else:
            model_name = "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/Qwen/Qwen2.5-VL-7B-Instruct"
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
    
    def inference(self, image: Image.Image, prompt: str) -> str:
        """
        Run inference on a single image to detect artifacts.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing artifact detection results
        """
        
        # Prepare the conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process the input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        # with torch.no_grad():
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # print(f"Generated output: {output_text}")
        # result = self._parse_response(output_text)

        return output_text

    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """
        Run inference on a batch of images to detect artifacts.

        Args:
            images: List of PIL Images to analyze

        Returns:
            List of dictionaries containing artifact detection results, one per image
        """
        if not images:
            return []

        # Build batched messages
        messages_list = []
        for image in images:
            messages_list.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            )

        # Build per-sample chat templates and vision inputs
        texts: List[str] = []
        batch_image_inputs: List[Any] = []
        for messages in messages_list:
            text = self.processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info([messages])
            texts.append(text)
            batch_image_inputs.append(image_inputs)

        # Tokenize/process as a batch
        inputs = self.processor(
            text=texts,
            images=batch_image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate batched outputs
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=512
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for output_text in output_texts:
            print(output_text)

        return output_texts

class InternEval:
    """
    Wrapper class for Qwen2.5-VL model evaluation.
    
    This class provides a unified interface for running inference
    on images to detect and describe artifacts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Qwen model for evaluation.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_finetuned = config.get('use_finetuned', False)
        
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the InternVL2 / InternVL3 model and processor."""
        # model_name = "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/OpenGVLab/InternVL2-8B"
        model_name = "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/OpenGVLab/InternVL3-8B"
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    def inference(self, image: Image.Image, prompt: str) -> str:
        """
        Run inference on a single image to detect artifacts.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing artifact detection results
        """
        
        transform = self._build_transform(input_size=448)
        images = self._dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # Use the model's built-in chat method for simplicity and robustness
        generation_config = dict(max_new_tokens=512, do_sample=False)
        
        # Call the model's chat method directly
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                prompt, 
                generation_config
            )
        
        return response
        
    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """
        Run inference on a batch of images to detect artifacts.

        Args:
            images: List of PIL Images to analyze

        Returns:
            List of dictionaries containing artifact detection results, one per image
        """
        if not images:
            return []

        if prompt is None:
            prompt = create_prompt('legion')

        results = []
        # Process images individually for now (batch processing can be complex with InternVL)
        for image in images:
            try:
                result = self.inference(image, prompt)
                results.append(result)
            except Exception as e:
                results.append({
                    "raw_response": "",
                    "error": str(e)
                })

        return results
    
    def _build_transform(self, input_size):
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

class GPTEval:
    """
    Wrapper class for GPT model evaluation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load model and processor
        self._init_client()
        self.money_manager = MoneyManager(model="gpt-4o")

    def _init_client(self):
        """Initialize openai client with API key"""
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        
        self.client = openai.OpenAI()

    def _create_prompt(self) -> str:
        """
        Create the prompt for artifact detection.
        
        Returns:
            Formatted prompt string for the model
        """
        return "Analyze the image and describe any visual anomalies. Provide whether there is an artifact, and if so, provide bboxes and descriptions for all anomalies. Respond with a JSON array of these objects in the following structured format: ```json\n[\n    {\n \"number_of_artifacts\": num,\n    \"artifacts\":\n[\n {\n        \"bbox_2d\": [x_min, y_min, x_max, y_max],\n        \"explanation\": \"The image contains an artifact of type ... on the ... of the ....\"\n    }, ...\n]\n}\n]\n```"

    def _encode_image_to_base64(self, image):
        """Convert PIL Image or numpy array to base64 string"""
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def inference(self, image: Image.Image, prompt: str) -> str:
        base64_image = self._encode_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.2
            )

            if self.money_manager:
                self.money_manager(response)

            raw_text = response.choices[0].message.content.strip()
            return raw_text

        except Exception as e:
            print(f"Error analyzing sampled instance: {e}")
            return None

    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """Batch inference by looping per image (API-friendly)."""
        results: List[Dict[str, Any]] = []
        for img in images:
            try:
                res = self.inference(img, prompt)
            except Exception as e:
                res = {
                    "error": str(e),
                }
            if res is None:
                res = {"number_of_artifacts": 0, "artifacts": []}
            results.append(res)
        return results

class GeminiEval:
    """
    Wrapper class for Gemini model evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load model and processor
        try:
            self._init_client()
        except Exception as e:
            print(f"Exception occurred while setting up Gemini client: {e}")
            self.client = None

        self.money_manager = MoneyManager(model="gemini-2.5-pro")

    def _init_client(self, service_account_path: str = "key/gemini_gcp.json",
        project_id: str = "gamebench-456108",
        location: str = "us-central1"):
        """Initialize gemini client with API key"""
        
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path, scopes=scopes
        )
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials,
        )
        self.client = client

    def _chat_completion_request(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.8,
        max_tokens: int = 1024,
        stream: bool = False,
        response_format: str = None,
    ):

        system_prompt = None
        contents = []

        for m in messages:
            if m["role"] == "system" and system_prompt is None:
                system_prompt = m["content"]
            else:
                if isinstance(m["content"], str):
                    contents.append(
                        types.Content(
                            role=m["role"], parts=[types.Part.from_text(text=m["content"])]
                        )
                    )
                elif isinstance(m["content"], list):
                    for part in m["content"]:
                        if part["type"] == "text":
                            contents.append(
                                types.Content(
                                    role=m["role"],
                                    parts=[types.Part.from_text(text=part["text"])],
                                )
                            )
                        elif part["type"] == "image_url":
                            base64_image = part["image_url"]["url"][
                                len("data:image/png;base64,") :
                            ]
                            image_bytes = base64.b64decode(base64_image)
                            contents.append(
                                types.Content(
                                    role=m["role"],
                                    parts=[
                                        types.Part.from_bytes(
                                            data=image_bytes, mime_type="image/png"
                                        )
                                    ],
                                )
                            )
                        else:
                            raise ValueError("Content must be a string or list of strings.")

        if system_prompt is None:
            system_prompt = ""

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            response_modalities=["TEXT"],
            safety_settings=[],
            system_instruction=[types.Part(text=system_prompt)],
        )

        if response_format:
            generate_content_config.response_mime_type = "application/json"
            generate_content_config.response_schema = response_format

        full_text = ""
        response_role = ""  # default role

        if stream:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    full_text += chunk.text
                    response_role = "assistant"
        else:
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    )

                    found_text = False
                    if hasattr(response, "candidates") and response.candidates:
                        for candidate in response.candidates:
                            if candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, "text") and part.text:
                                        found_text = True
                                        break
                            if found_text:
                                break
                    if found_text:
                        break

                    print(f"[Retry {attempt+1}] Empty response. Retrying...")
                except Exception as e:
                    import time

                    print(f"[Retry {attempt + 1}] Unexpected error: {e}. Retrying...")
                    time.sleep(2**attempt)

        return response


    def inference(self, image: Image.Image, prompt: str) -> str:
        if self.client is None:
            return {"error": "gemini_client_not_initialized"}
        try:
            # Upload image bytes
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[img_part, "\n\n", prompt],
                config=types.GenerateContentConfig(max_output_tokens=65535, temperature=0.2, top_p=0.8),
            )
            if self.money_manager:
                self.money_manager(response)

            # Extract text from response
            raw_text = ""
            if hasattr(response, "candidates") and response.candidates:
                for cand in response.candidates:
                    if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                        for part in cand.content.parts:
                            if hasattr(part, "text") and part.text:
                                raw_text += part.text
                                
            raw_text = raw_text.strip()
            return raw_text
        except Exception as e:
            return {"error": str(e)}
    
    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        results: List[Dict[str, Any]] = []
        for img in images:
            results.append(self.inference(img, prompt))
        return results


class PalEval:
    """
    Wrapper class for PAL4VST model evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Multi-GPU configuration
        self.use_multi_gpu = config.get('use_multi_gpu', False)
        self.gpu_devices = self._setup_devices(config)
        self.device = self.gpu_devices[0] if self.gpu_devices else 'cpu'
    
        # Load model and processor
        self._load_model()
        
    def _setup_devices(self, config: Dict[str, Any]) -> List[str]:
        """
        Setup available GPU devices for multi-GPU inference.
        """
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return ['cpu']
            
        # Check if multi-GPU is requested
        if not config.get('use_multi_gpu', False):
            device = config.get('device', 'cuda:0')
            return [device]
            
        # Get available GPU devices
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"Only {num_gpus} GPU available, falling back to single GPU")
            return [config.get('device', 'cuda:0')]
            
        # Use specified devices or all available
        specified_devices = config.get('gpu_devices', None)
        if specified_devices:
            # Validate specified devices
            valid_devices = []
            for device in specified_devices:
                if isinstance(device, int):
                    device = f'cuda:{device}'
                if device.startswith('cuda:'):
                    gpu_id = int(device.split(':')[1])
                    if gpu_id < num_gpus:
                        valid_devices.append(device)
                    else:
                        print(f"GPU {gpu_id} not available, skipping")
            return valid_devices if valid_devices else ['cuda:0']
        else:
            # Use first two GPUs by default
            devices = [f'cuda:{i}' for i in range(min(2, num_gpus))]
            print(f"Using GPUs: {devices}")
            return devices

    def _load_model(self):
        """Load the PAL4VST torchscript."""
        torchscript_file = "/home/jovyan/image-artifacts/baselines/PAL4VST/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt"
            
        if self.use_multi_gpu and len(self.gpu_devices) > 1:
            # Load model replicas on each GPU
            self.models = {}
            for device in self.gpu_devices:
                print(f"Loading PAL model on {device}")
                self.models[device] = torch.jit.load(torchscript_file).to(device)
            
            # Keep primary model reference for single inference
            self.model = self.models[self.device]
            print(f"Multi-GPU setup complete with {len(self.gpu_devices)} GPUs: {self.gpu_devices}")
        else:
            # Single GPU setup
            self.model = torch.jit.load(torchscript_file).to(self.device)
            self.models = {self.device: self.model}
    
    def inference(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single image to detect artifacts.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            (512, 512) sized heatmap of artifact
        """
        img_tensor = self._prepare_input(np.array(image.resize((512, 512))))

        result = self.model(img_tensor).cpu().data.numpy()[0][0]
        return {"heatmap": result}

    def inference_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        if not images:
            return []

        if self.use_multi_gpu and len(self.gpu_devices) > 1 and len(images) > 1:
            return self._inference_batch_multi_gpu(images)
        else:
            outputs: List[Dict[str, Any]] = []
            for img in images:
                try:
                    outputs.append(self.inference(img))
                except Exception as e:
                    outputs.append({"heatmap": None, "error": str(e)})
            return outputs
    
    def _inference_batch_multi_gpu(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Multi-GPU batch inference with load balancing.
        
        Args:
            images: List of PIL Images to analyze
            
        Returns:
            List of dictionaries containing heatmap results
        """
        import threading
        import queue
        
        num_gpus = len(self.gpu_devices)
        results = [None] * len(images)
        
        # Create task queue and result queue
        task_queue = queue.Queue()
        error_queue = queue.Queue()
        
        # Add tasks to queue with image index
        for i, image in enumerate(images):
            task_queue.put((i, image))
        
        def worker(device: str):
            """Worker function for each GPU."""
            model = self.models[device]
            
            while True:
                try:
                    # Get task from queue with timeout
                    img_idx, image = task_queue.get(timeout=1)
                except queue.Empty:
                    break
                
                try:
                    # Process image on this GPU
                    img_tensor = self._prepare_input(np.array(image.resize((512, 512))), device)
                    img_tensor = img_tensor.to(device)
                    
                    with torch.no_grad():
                        heatmap = model(img_tensor).cpu().data.numpy()[0][0]
                    
                    results[img_idx] = {"heatmap": heatmap}
                    
                except Exception as e:
                    results[img_idx] = {"heatmap": None, "error": str(e)}
                    error_queue.put(f"Error on {device} for image {img_idx}: {e}")
                
                finally:
                    task_queue.task_done()
        
        # Start worker threads for each GPU
        threads = []
        for device in self.gpu_devices:
            thread = threading.Thread(target=worker, args=(device,))
            thread.start()
            threads.append(thread)
        
        # Wait for all tasks to complete
        task_queue.join()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=5)  # 5 second timeout per thread
        
        # Log any errors
        while not error_queue.empty():
            try:
                error_msg = error_queue.get_nowait()
                print(f"Multi-GPU inference error: {error_msg}")
            except queue.Empty:
                break
        
        # Fill any None results with error responses
        for i, result in enumerate(results):
            if result is None:
                results[i] = {"heatmap": None, "error": "Multi-GPU processing failed"}
        
        return results
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free up memory."""
        if torch.cuda.is_available():
            for device in self.gpu_devices:
                if device.startswith('cuda:'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
            print("GPU cache cleared for all devices")
    
    def get_gpu_memory_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get memory information for all GPUs.
        
        Returns:
            Dictionary with memory info for each GPU device
        """
        memory_info = {}
        if torch.cuda.is_available():
            for device in self.gpu_devices:
                if device.startswith('cuda:'):
                    gpu_id = int(device.split(':')[1])
                    memory_info[device] = {
                        'allocated_gb': torch.cuda.memory_allocated(gpu_id) / (1024**3),
                        'cached_gb': torch.cuda.memory_reserved(gpu_id) / (1024**3),
                        'max_allocated_gb': torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
                    }
        return memory_info
    
    def _get_mean_stdinv(self, img):
        """
        Compute the mean and std for input image (make sure it's aligned with training)
        """

        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]

        mean_img = np.zeros((img.shape))
        mean_img[:,:,0] = mean[0]
        mean_img[:,:,1] = mean[1]
        mean_img[:,:,2] = mean[2]
        mean_img = np.float32(mean_img)

        std_img = np.zeros((img.shape))
        std_img[:,:,0] = std[0]
        std_img[:,:,1] = std[1]
        std_img[:,:,2] = std[2]
        std_img = np.float64(std_img)

        stdinv_img = 1 / np.float32(std_img)

        return mean_img, stdinv_img

    def _numpy2tensor(self, img):
        """
        Convert numpy to tensor
        """
        img = torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()
        return img

    def _prepare_input(self, img, device=None):
        """
        Convert numpy image into a normalized tensor (ready to do segmentation)
        """
        if device is None:
            device = self.device

        mean_img, stdinv_img = self._get_mean_stdinv(img)

        img_tensor = self._numpy2tensor(img).to(device)
        mean_img_tensor = self._numpy2tensor(mean_img).to(device)
        stdinv_img_tensor = self._numpy2tensor(stdinv_img).to(device)
        
        img_tensor = img_tensor - mean_img_tensor
        img_tensor = img_tensor * stdinv_img_tensor

        return img_tensor


class DiffEval:
    """
    DiffDoctor artifact detector wrapper.
    Outputs heatmap of shape (1, 1, 512, 512) under key 'heatmap'.
    Reference: https://github.com/ali-vilab/DiffDoctor.git
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _get_segformer(self, path_or_hub, out_channels=1):
        # load a pretrained Segformer model
        self.preprocessor = SegformerImageProcessor.from_pretrained(path_or_hub)
        model = SegformerForSemanticSegmentation.from_pretrained(path_or_hub)
        # change the number of output channels
        model.decode_head.classifier = torch.nn.Conv2d(model.decode_head.classifier.in_channels, out_channels, kernel_size=1)
        return model

    def _load_model(self) -> None:
        base_dir = "/home/jovyan/image-artifacts/baselines/DiffDoctor"
        ckpt = os.path.join(base_dir, "checkpoints", "ad_pytorch_model.bin")
        self.model = self._get_segformer("nvidia/mit-b5", out_channels=1)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.to(self.device)
        self.model.eval()

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        if self.model is None:
            return {"heatmap": None, "error": "diffdoctor_model_not_loaded"}
        with torch.no_grad():
            image = transforms.ToTensor()(image).to(self.device)
            x = self.preprocessor(image, return_tensors='pt',do_rescale=False)['pixel_values'].to(self.device)
            pred = self.model(x)
            pred = torch.nn.functional.interpolate(
                pred.logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            out = torch.sigmoid(pred)
        if isinstance(out, torch.Tensor):
            out_np = out.detach().cpu().numpy()
        else:
            out_np = np.array(out)
        if out_np.ndim == 2:
            out_np = out_np[None, None, ...]
        elif out_np.ndim == 3:
            out_np = out_np[None, ...]
        return {"heatmap": out_np}

    def inference_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        return [self.inference(img) for img in images]
   