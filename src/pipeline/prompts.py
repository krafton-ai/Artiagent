import json
from PIL import Image
import base64
import io
import numpy as np
import re
import os
import cv2
from openai.types.chat import ChatCompletion
from typing import Union

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

        else:  # OpenAI and Claude
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens 

            if "o1" in self.model or "o3" in self.model or "o4" in self.model:
                output_tokens += (
                    response.usage.completion_tokens_details.accepted_prediction_tokens
                    + response.usage.completion_tokens_details.reasoning_tokens
                    + response.usage.completion_tokens_details.rejected_prediction_tokens
                )

        input_cost = input_tokens / 1000 * self.input_cost
        output_cost = output_tokens / 1000 * self.output_cost

        self.total_cost += input_cost + output_cost

    def refresh(self) -> None:
        self.total_cost = 0.0

def get_entity_subentities(client, image, money_manager=None):
    base64_image = encode_image_to_base64(image)
    system_prompt = """
    You are given an image. Identify the visible **entities** and their **subentities**.

    **Return ONLY JSON (no prose, no code fences).** Output must be a single JSON object that maps entities to their subentities, using this schema:

    {"<entity>": ["<sub1>", "<sub2>", "..."], "<entity2>": ["<sub1>", "<sub2>", "..."], ...}

    Hard rules:
    1) Each returned entity MUST have **at least one** subentity. **Never** return an empty list for any entity.
    2) If you cannot name at least one clearly visible subentity for an entity, **omit that entity entirely** (do not list it at all).
    3) Subentities must be **clearly visible** and **reasonably segmentable** in the image.
    4) **Exclude** parts that are tightly bound to or visually fused with the torso/core body (e.g., arms pressed to sides, folded wings against body). Only include subentities with clear visual separation.
    5) Do **not** invent parts that are occluded, cropped out, or ambiguous.
    6) Use concise, lowercase **nouns**; de-duplicate terms. Prefer 1–6 subentities per entity.
    7) **Return exactly one JSON object** (not an array, not multiple separate objects).
    8) **Granularity rule (coarsity):** Choose the **most specific visible entity**. If only a part is clearly visible (e.g., a leg without enough evidence of the full person), output that part as the entity (e.g., "leg") rather than its parent (e.g., "person"). Do **not** infer parent entities that are not clearly visible.

    Clarifications:
    - Examples of valid subentities:
    • person → head, arm, hand, leg, foot, ear
    • hand → finger, nail, palm
    • dog → ear, snout, leg, tail, paw
    • car → wheel, door, window, mirror
    - Avoid generic torso-like regions. If no fine-grained parts are clearly separable for a candidate entity, **omit the entity** rather than returning an empty list.
    - Granularity examples:
    • If only a single **leg** is clearly visible: {"leg": ["knee", "ankle", "foot"]}  # Good
    • Not acceptable for the same case: {"person": ["leg"]}  # Bad (parent entity not sufficiently visible)

    Bad examples (NOT allowed):
    {"dog": []}  # empty subentities list
    {"suitcase": ["handle"]}, {"chair": ["leg"]}  # multiple separate JSON objects

    Good examples:
    {"dog": ["ear", "leg", "head"]}

    {
    "person": ["head", "arm", "hand", "leg", "foot"],
    "hand": ["finger", "nail", "palm"]
    }
    """

    response = client.chat.completions.create(
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

    # Track costs with money manager
    if money_manager:
        money_manager(response)

    response = response.choices[0].message.content.strip()
        # Try to extract JSON from the response
    try:
        # First, try to parse the entire response as JSON
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # If that fails, try to find JSON within the response
        import re
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing fails, return the raw text for debugging
        print(f"Could not parse JSON from response: {response}")
        return {
            "error": "json_parse_failed",
            "raw_response": response
        }


def encode_image_to_base64(image):
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

def caption_image_with_openai(client, image, money_manager=None):
    """Generate caption for image using OpenAI Vision API"""
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed caption describing this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # Track costs with money manager
        if money_manager:
            money_manager(response)
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None

def query_addition_artifact_success(client, img_array, mask_image, part_entity_name, money_manager=None):
    """
    Query GPT-4 Vision to check if addition artifact injection was successful.
    
    Args:
        client: OpenAI client
        img_array: Original image as numpy array
        mask_image: PIL Image of the target mask region
        part_entity_name: Description of the part entity (e.g., "a hand of a person")
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        dict: Contains 'success' boolean and 'reasoning' string
    """
    # Encode images to base64
    base64_image = encode_image_to_base64(img_array)
    base64_mask = encode_image_to_base64(mask_image)
    
    prompt = f"""
    You are an expert at detecting addition-type artifacts in AI-generated images.

    Addition artifacts occur when a part of an object is duplicated and placed adjacent to the original, 
    creating anatomically or structurally implausible duplications (e.g., extra fingers, duplicate ears, etc.).

    You will be shown:
    1. An original image
    2. A mask highlighting a specific region of interest
    
    Your task is to determine if there is "{part_entity_name}" present in the masked region of the image.
    
    Look carefully at the masked region and determine:
    - Is there a clear presence of "{part_entity_name}" within the highlighted area?
    - Does it appear to be a plausible duplication/addition of the specified part?
    
    Consider that addition artifacts should:
    - Show the specified part type in the masked region
    - Appear anatomically/structurally similar to other instances of the same part
    - Be positioned adjacent to where such parts would naturally occur
    
    Return your analysis in this exact JSON format:
    {{
        "success": true/false,
        "reasoning": "Brief explanation of what you observe in the masked region"
    }}
    
    Set "success" to true if you can clearly identify "{part_entity_name}" in the masked region.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_mask}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        # Track costs with money manager
        if money_manager:
            money_manager(response)
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Fallback parsing
                return {
                    "success": False,
                    "confidence": 0.0,
                    "reasoning": "Could not parse response",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            return {
                "success": False,
                "confidence": 0.0, 
                "reasoning": "JSON parsing failed",
                "raw_response": response_text
            }
            
    except Exception as e:
        print(f"Error in addition artifact query: {e}")
        return {
            "success": False,
            "confidence": 0.0,
            "reasoning": f"API error: {str(e)}"
        }


def query_removal_artifact_success(client, img_array, mask_image, part_entity_name, money_manager=None):
    """
    Query GPT-4 Vision to check if removal artifact injection was successful.
    
    Args:
        client: OpenAI client
        img_array: Original image as numpy array
        mask_image: PIL Image of the target mask region
        part_entity_name: Description of the part entity (e.g., "a hand of a person")
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        dict: Contains 'success' boolean and 'reasoning' string
    """
    # Encode images to base64
    base64_image = encode_image_to_base64(img_array)
    base64_mask = encode_image_to_base64(mask_image)
    
    prompt = f"""
    You are an expert at detecting removal-type artifacts in AI-generated images.

    Removal artifacts occur when a part of an object is deleted and the area is inpainted with background, 
    resulting in missing parts that should be present (e.g., missing fingers, absent ears, etc.).

    You will be shown:
    1. An original image  
    2. A mask highlighting a specific region of interest
    
    Your task is to determine if there is NO "{part_entity_name}" present in the masked region of the image.
    
    Look carefully at the masked region and determine:
    - Is the specified part clearly absent from the highlighted area?
    - Does the region show signs of inpainting or background fill instead of the expected part?
    
    Consider that successful removal artifacts should:
    - Show absence of the specified part in the masked region
    - Display background textures or inpainting in place of the missing part
    - Leave the surrounding anatomy/structure intact but incomplete
    
    Return your analysis in this exact JSON format:
    {{
        "success": true/false,
        "reasoning": "Brief explanation of what you observe in the masked region"
    }}
    
    Set "success" to true if you can confirm that "{part_entity_name}" is clearly absent from the masked region.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_mask}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        # Track costs with money manager
        if money_manager:
            money_manager(response)
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Fallback parsing
                return {
                    "success": False,
                    "confidence": 0.0,
                    "reasoning": "Could not parse response",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            return {
                "success": False,
                "confidence": 0.0,
                "reasoning": "JSON parsing failed", 
                "raw_response": response_text
            }
            
    except Exception as e:
        print(f"Error in removal artifact query: {e}")
        return {
            "success": False,
            "confidence": 0.0,
            "reasoning": f"API error: {str(e)}"
        }

def artifact_explanation(client, real_image, artifact_image, entity, part, artifact_type, money_manager=None):
    """
    Generate natural language explanation of visual artifacts using OpenAI Vision API
    
    Args:
        client: OpenAI client
        real_image: Original image as numpy array with region visualized where artifact will be injected
        artifact_image: Modified image with artifact as numpy array with region visualized where artifact was injected
        metadata: Dictionary containing:
            - 'entity': e.g., 'person'
            - 'part': e.g., 'left leg'
            - 'artifact_type': e.g., 'distortion' (for reasoning only)
            - 'target_bbox': bounding box coordinates [x1, y1, x2, y2]
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        dict: Contains 'explanation' string and 'success' boolean
    """
    # Encode both images to base64
    base64_real_image = encode_image_to_base64(real_image)
    base64_artifact_image = encode_image_to_base64(artifact_image)
    
    # Create artifact-type-specific guidance (without explicitly mentioning the type)
    if artifact_type == 'distortion':
        focus_guidance = "Pay attention to warped shapes, unnatural geometry, irregular textures, or visual blending errors that make the structure appear broken or malformed."
    elif artifact_type == 'removal':
        focus_guidance = "Look for missing structure, unnatural gaps, smoothed-over areas, or anatomical discontinuity where something appears to be absent."
    elif artifact_type == 'addition':
        focus_guidance = "Notice any duplicated or misplaced parts, unnatural growths, or extra elements that conflict with normal anatomy or structure."
    else:
        focus_guidance = "Identify any visual abnormalities, unnatural features, or elements that appear incorrect or implausible."
    
    prompt = f"""
You are given two images:

- **Image A**: A real, original image, with a region visualized as red bounding box where the artifact is going to be injected.
- **Image B**: A modified version of the same scene, with a region visualized as green bounding box where the artifact is injected.

Here is the structured context:
- **Entity**: {entity}
- **Part**: {part}

Your task is to:
1. Examine the highlighted region in the **given image** (Image B).
2. Use your understanding of how the specified part of the entity should normally appear to identify abnormalities.
3. Write a natural language explanation describing what appears visually wrong or unnatural in the highlighted region.

**Do not mention or refer to the original image, the artifact type, or the image source explicitly.**
Base your explanation only on what is visible in the given image.

Focus your reasoning based on the artifact type (without stating it):

{focus_guidance}

Respond naturally and precisely, describing only what is visibly incorrect in the given image within the highlighted region.

Return your response in this exact JSON format:
{{
    "explanation": "Your detailed explanation of what appears visually wrong in the highlighted region"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_real_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_artifact_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        # Track costs with money manager
        if money_manager:
            money_manager(response)
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "success": True,
                    "explanation": result.get("explanation", ""),
                    "raw_response": response_text
                }
            else:
                # Fallback: use the entire response as explanation
                return {
                    "success": True,
                    "explanation": response_text,
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            # Fallback: use the entire response as explanation
            return {
                "success": True,
                "explanation": response_text,
                "raw_response": response_text
            }
            
    except Exception as e:
        print(f"Error in artifact explanation: {e}")
        return {
            "success": False,
            "explanation": "",
            "error": f"API error: {str(e)}"
        } 
        

########## deprecated ##########
'''
def get_entity_subparts_by_type(client, image, artifact_type, money_manager=None):

    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    artifact_type_prompt = {
        'addition': """
        You are an image artifact agent, where you will decide which part of an object present in the image to inject an addition-type artifact.

        Given an image, select an entity suitable for artifact injection, and generate candidate entity parts that are suitable for generating addition-type artifacts.

        Do **not** include parts that are tightly overlapping or fused with the entity's torso or core body. For example, if a bird's wings are folded closely against its torso, they should not be selected. In such cases, the addition artifact may appear as if it's modifying the torso itself, which is not appropriate.

        Return the candidates in JSON format. Example:

            {
            "entity": "person",
            "subparts": ["arm", "hand", "leg", "foot", "eye"]
            }

            {
            "entity": "car",
            "subparts": ["mirror", "wheel"]
            }
        """,
        'removal': """
        You are an image artifact agent, where you will decide which part of an object present in the image to inject a removal-type artifact.

        Given an image, select an entity suitable for artifact injection, and generate candidate entity parts that are suitable for generating removal-type artifacts.

        Do **not** include parts that are tightly overlapping or fused with the entity's torso or core body. For example, if a bird’s wings are folded closely against its body, removing them would resemble torso removal, which is not a valid removal artifact.

        Return the candidates in JSON format. Example:

            {
            "entity": "person",
            "subparts": ["arm", "hand", "leg", "foot", "eye"]
            }

            {
            "entity": "car",
            "subparts": ["mirror", "wheel"]
            }
        """,
        'distortion': """
        You are an image artifact agent, where you will decide which part of an object present in the image to inject distortion type of artifact.

        Given an image, select an entity suitable for artifact injection, and generate candidate of entity parts that is suitable for generating distortion type of artifact.

        Return the candidiates in json format. I will provide you with some output example format.

            {
            "entity": "person",
            "subparts": ["leg", "head"]
            }
            
            {
            "entity": "car",
            "subparts": ["mirror", "wheel", "door"]
            }
        """
    }
    
    system_prompt = """
    Image artifacts refer to unintended, implausible, or visibly corrupted regions within images generated by diffusion models. These artifacts often break the natural semantics or visual coherence of an image, such as a person with extra fingers, a car with warped wheels, or missing parts of animals, and can significantly degrade image quality or realism. Artifacts are a critical concern in both model evaluation and training.

    There are three types of image artifacts: Addition, Removal, and Distortion.

    1. Addition: Involves duplicating an existing part of the image and placing it somewhere else, creating implausible duplication (e.g., extra thumb, leg, or ear). The added part is placed adjacent to the original, in one of four directions. 
        - Common on peripheral or terminal parts of objects/entities:
            - Human/Animal: fingers, hands, toes, ears, etc.
            - Vehicles: mirrors, wheels, wipers.

    2. Removal: A specific object or part is deleted, and the area is inpainted using background textures, resulting in missing limbs, features, or objects, sometimes with visible traces.
        - Common on terminal or protruding elements:
            - Human/Animal: fingers, toes, legs, ears, horns, etc.
            - Vehicles: antennas, side mirrors, etc.

    3. Distortion: The object or part remains in place but the structure is altered (e.g., twisted, warped, scrambled), making the object unrecognizable or visually broken, like a warped face or twisted wheel.
        - Can occur anywhere, especially in central or continuous regions:
            - Human/Animal: face, arm, leg, etc.
            - Vehicles: car doors, etc.

    Important constraint:
    - You must **only recommend parts that are clearly visible in the image**. Do **not** include parts that are occluded, cropped out, or ambiguous. Artifact injection should only be applied to parts that are identifiable and visually distinguishable.
    """ + artifact_type_prompt[artifact_type] + """

    ### Output:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.2
        )

        # Track costs with money manager
        if money_manager:
            money_manager(response)

        response = response.choices[0].message.content.strip()
        # Try to extract JSON from the response
        try:
            # First, try to parse the entire response as JSON
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the response
            import re
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, return the raw text for debugging
            print(f"Could not parse JSON from response: {response}")
            return {
                "error": "json_parse_failed",
                "raw_response": response
            }

    except Exception as e:
        print(f"Error analyzing sampled instance: {e}")
        return None

def get_all_entity_subparts(client, image, money_manager=None):
    base64_image = encode_image_to_base64(image)

    system_prompt = """
    You are an image artifact agent. Image artifacts refer to unintended, implausible, or visibly corrupted regions within images generated by diffusion models. These artifacts often break the natural semantics or visual coherence of an image, such as a person with extra fingers, a car with warped wheels, or missing parts of animals, and can significantly degrade image quality or realism.
    
    Your task is to analyze the image and output **visible entities and their suitable subparts** for three types of artifact injection: **addition**, **removal**, and **distortion**.

    ---

    ### 1. Addition: Involves duplicating an existing part of the image and placing it somewhere else, creating implausible duplication (e.g., extra thumb, leg, or ear). The added part is placed adjacent to the original, in one of four directions. 
        - Common on peripheral or terminal parts of objects/entities:
            - Human/Animal: fingers, hands, toes, legs, etc.
            - Vehicles: mirrors, wheels, wipers.
        - Constraint: Do **not** include parts that are tightly overlapping or fused with the entity's torso or core body. For example, if a bird's wings are folded closely against its torso, they should not be selected. In such cases, the addition artifact may appear as if it's modifying the torso itself, which is not appropriate.
    ---

    ### 2. Removal: A specific object or part is deleted, and the area is inpainted using background textures, resulting in missing limbs, features, or objects, sometimes with visible traces.
        - Common on terminal or protruding elements:
            - Human/Animal: fingers, toes, legs, ears, horns, etc.
            - Vehicles: antennas, side mirrors, etc.
        - Constraint: Do **not** include parts that are tightly overlapping or fused with the entity's torso or core body. For example, if a bird’s wings are folded closely against its body, removing them would resemble torso removal, which is not a valid removal artifact.

    ---

    ### 3. Distortion: The object or part remains in place but the structure is altered (e.g., twisted, warped, scrambled), making the object unrecognizable or visually broken, like a warped face or twisted wheel.
        - Can occur anywhere, especially in central or continuous regions:
            - Human/Animal: face, torso, entire leg, etc.
            - Vehicles: car doors, etc.

    ---

    Important constraint:
    - You must **only recommend parts that are clearly visible in the image**. Do **not** include parts that are occluded, cropped out, or ambiguous. Artifact injection should only be applied to parts that are identifiable and visually distinguishable.

    Return the results in **JSON** format. I will provide you with some examples:

    ```json
    {
    "addition": {
        "entity": "giraffe",
        "subparts": ['ear', 'leg']
    },
    "removal": {
        "entity": "giraffe",
        "subparts": ['ear', 'horn', 'leg']
    },
    "distortion": {
        "entity": "dog",
        "subparts": ['face', 'torso', 'legs']
    }
    }
    ```

    ```json
    {
    "addition": {
        "entity": "hand",
        "subparts": ['finger', 'thumb']
    },
    "removal": {
        "entity": "hand",
        "subparts": ['finger', 'thumb']
    },
    "distortion": {
        "entity": "hand",
        "subparts": ['fingers', 'palm']
    }
    }
    ```

    ### Output:
    """

    try:
        response = client.chat.completions.create(
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

        if money_manager:
            money_manager(response)

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





'''
########## deprecated ##########