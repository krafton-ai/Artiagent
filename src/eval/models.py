"""
Model classes for artifact detection evaluation.

This module contains model wrappers for different VLM/MLLM models
to evaluate their artifact detection capabilities.
"""

import torch
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import json
import re
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import openai
from openai.types.chat import ChatCompletion

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
            # TODO: cost changes when # tokens > 200k
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
            model_name = "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/saves/qwen2_5vl-7b/full/sft_artifacts"
        else:
            model_name = "/home/jovyan/image-artifacts/src/train/LLaMA-Factory/Qwen/Qwen2.5-VL-7B-Instruct"
            
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def _create_prompt(self) -> str:
        """
        Create the prompt for artifact detection.
        
        Returns:
            Formatted prompt string for the model
        """

        prompt = "Analyze the image and describe any visual anomalies. Provide whether there is an artifact, and if so, provide bboxes and descriptions for all anomalies. Respond with a JSON array of these objects in the following structured format: ```json\n[\n    {\n \"number_of_artifacts\": num,\n    \"artifacts\":\n[\n {\n        \"bbox_2d\": [x_min, y_min, x_max, y_max],\n        \"explanation\": \"The image contains an artifact ... on the ... of the ....\"\n    }, ...\n]\n}\n]\n```"
        return prompt
    
    def inference(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single image to detect artifacts.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing artifact detection results
        """
        prompt = self._create_prompt()
        
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
            result = {
                "number_of_artifacts": 0,
                "artifacts": [],
                "raw_response": output_text,
                "error": str(e)
            }
            
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

        prompt = self._create_prompt()

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

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        prompt = self._create_prompt()
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
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

                return json.loads(raw_text)

            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*?\}', raw_text)
                if json_match:
                    try:
                        return json.loads(json_match.group())
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


class GeminiEval:
    """
    Wrapper class for Gemini model evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load model and processor
        self._init_client()
        self.money_manager = MoneyManager(model="gemini-2.5-pro")

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

    def inference(self, image: Image.Image) -> Dict[str, Any]:
        prompt = self._create_prompt()
        try:
            response = self.client.chat.completions.create(
                model="gemini-2.5-pro",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
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

                return json.loads(raw_text)

            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*?\}', raw_text)
                if json_match:
                    try:
                        return json.loads(json_match.group())
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
