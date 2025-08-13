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
from qwen_vl_utils import process_vision_info
import openai
from openai.types.chat import ChatCompletion
from google.oauth2 import service_account
from google import genai
from google.genai import types

def create_prompt(option):
    if option == 'default':
        prompt = "Analyze the image and describe any visual anomalies. Provide whether there is an artifact, and if so, provide bboxes and descriptions for all anomalies. Respond with a JSON array of these objects in the following structured format: ```json\n[\n    {\n \"number_of_artifacts\": num,\n    \"artifacts\":\n[\n {\n        \"bbox_2d\": [x_min, y_min, x_max, y_max],\n        \"explanation\": \"The image contains an artifact of type ... on the ... of the ....\"\n    }, ...\n]\n}\n]\n```"
    elif option == 'legion':
        # prompt = "Please provide a detailed analysis of this photo, considering the following: Physics artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), Structure artifacts (e.g., deformed objects, asymmetry, or distorted text), and Distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved bboxes for the corresponding parts of the answer in the following structured format: ```json\n[\n    {\n \"number_of_artifacts\": num,\n   \"caption\": \"Upon examining the image, I have found: ... .\",\n   \"artifacts\":\n[\n {\n        \"bbox_2d\": [x_min, y_min, x_max, y_max],\n        \"explanation\": \"... .\"\n    }, ...\n]\n}\n]\n```"
        prompt = "You are a helpful assistant. Analyze the given images based on the following three criteria and return the visible artifacts in the image. You need to return the bbox and explanations for all visible artifacts, and if none, assign 0 to the number of artifacts. Evaluation Criteria: (a) The image should be well-lit, sharp, and visually clear without blurriness, noise, or distortion. (b) The image must not show obvious signs of artificial manipulation, such as pixelated edges or unnatural distortions. (c) The image should look realistic and have a photo-like appearance. Localization Task : Return all the visible artifacts in the structured JSON format:  ```json\n[\n    {\n \"number_of_artifacts\": num,\n    \"artifacts\":\n[\n {\n        \"bbox_2d\": [x_min, y_min, x_max, y_max],\n        \"explanation\": \"... .\"\n    }, ...\n]\n}\n]\n``` Please strictly follow the instructions to label the input image."
    elif option == 'synartifact':
        prompt = "Step 1: You are my assistant to analyze whether artifacts exist in this image. If there are any artifacts, go to step 2. If not, go to step 5. Step 2: You are my assistant to locate artifacts in this image. Please provide the coordinates for artifacts that you choose using the format of [x1,y1,x2,y2]. Step 3: You are my assistant to explain anomalies in this image. Please provide detailed explanations of the artifact in the bbox you have selected in step 2. Step 4: You are my assistant to analyze other artifacts in this image. If there are any other artifacts except the above in this image, go back to step 2 and repeat. If not, go to step 5. Step 5: Gather all the information above and return the JSON output in the structured format: ```json\n[\n    {\n \"number_of_artifacts\": num,\n    \"artifacts\":\n[\n {\n        \"bbox_2d\": [x_min, y_min, x_max, y_max],\n        \"explanation\": \"The image contains an artifact of type ... on the ... of the ....\"\n    }, ...\n]\n}\n]\n```"

    return prompt

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
    
    def inference(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single image to detect artifacts.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing artifact detection results
        """
        prompt = create_prompt('legion')
        
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
        
        # Parse the JSON response
        try:
            result = self._parse_response(output_text)
        except Exception as e:
            # Fallback if JSON parsing fails
            try:
                output_text = output_text.replace("addCriterion", "")
                result = self._parse_response(output_text)
            except Exception as e:
                result = {
                    "number_of_artifacts": 0,
                    "artifacts": [],
                    "raw_response": output_text,
                    "error": str(e)
                }
        print(f"Generated output: {output_text}")
        return result

    def inference_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images to detect artifacts.

        Args:
            images: List of PIL Images to analyze

        Returns:
            List of dictionaries containing artifact detection results, one per image
        """
        if not images:
            return []

        prompt = create_prompt('synartifact')

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

        results: List[Dict[str, Any]] = []
        for output_text in output_texts:
            try:
                results.append(self._parse_response(output_text))
            except Exception as e:
                results.append(
                    {
                        "number_of_artifacts": 0,
                        "artifacts": [],
                        "raw_response": output_text,
                        "error": str(e),
                    }
                )
            print(output_text)

        return results
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the model's response to extract artifact information.
        
        Args:
            response: Raw text response from the model
            
        Returns:
            Parsed dictionary with artifact information
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                pass
        
        # If no JSON found, return empty result
        return {
            "number_of_artifacts": 0,
            "artifacts": [],
            "raw_response": response
        }


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

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        prompt = create_prompt('legion')
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

            try:
                raw_text = response.choices[0].message.content.strip()

                # Handle markdown-style code block like ```json ... ```
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text[len("```"):].strip()
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

                return json.loads(raw_text)[0]

            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*?\}', raw_text)
                if json_match:
                    try:
                        return json.loads(json_match.group())[0]
                    except json.JSONDecodeError:
                        pass

                print(f"Could not parse JSON from response: {raw_text}")
                return {
                    "error": "json_parse_failed",
                    "raw_response": raw_text
                }

        except Exception as e:
            print(f"Error analyzing sampled instance: {e}")
            return None

    def inference_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Batch inference by looping per image (API-friendly)."""
        results: List[Dict[str, Any]] = []
        for img in images:
            try:
                res = self.inference(img)
            except Exception as e:
                res = {
                    "number_of_artifacts": 0,
                    "artifacts": [],
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


    def inference(self, image: Image.Image) -> Dict[str, Any]:
        if self.client is None:
            return {"number_of_artifacts": 0, "artifacts": [], "error": "gemini_client_not_initialized"}
        prompt = create_prompt('legion')
        try:
            # Upload image bytes
            buf = io.BytesIO()
            image.convert("PNG").save(buf, format="PNG")
            buf.seek(0)
            img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[img_part, "\n\n", prompt],
                config=types.GenerateContentConfig(max_output_tokens=1024, temperature=0.2, top_p=0.8),
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

            try:
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text[len("```"):].strip()
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()
                parsed = json.loads(raw_text)
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]
                return parsed
            except json.JSONDecodeError:
                json_match = re.search(r"\{[\s\S]*?\}", raw_text)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        if isinstance(parsed, list) and parsed:
                            parsed = parsed[0]
                        return parsed
                    except json.JSONDecodeError:
                        pass
                return {"number_of_artifacts": 0, "artifacts": [], "raw_response": raw_text, "error": "json_parse_failed"}
        except Exception as e:
            return {"number_of_artifacts": 0, "artifacts": [], "error": str(e)}
    
    def inference_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for img in images:
            results.append(self.inference(img))
        return results


class PalEval:
    """
    Wrapper class for PAL4VST model evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the PAL4VST torchscript."""
        torchscript_file = "/home/jovyan/image-artifacts/baselines/PAL4VST/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt"
            
        self.model = torch.jit.load(torchscript_file).to(self.device)
    
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
        outputs: List[Dict[str, Any]] = []
        for img in images:
            try:
                outputs.append(self.inference(img))
            except Exception as e:
                outputs.append({"heatmap": None, "error": str(e)})
        return outputs
    
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

    def _prepare_input(self, img):
        """
        Convert numpy image into a normalized tensor (ready to do segmentation)
        """
        mean_img, stdinv_img = self._get_mean_stdinv(img)

        img_tensor = self._numpy2tensor(img).to(self.device)
        mean_img_tensor = self._numpy2tensor(mean_img).to(self.device)
        stdinv_img_tensor = self._numpy2tensor(stdinv_img).to(self.device)
        
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
   