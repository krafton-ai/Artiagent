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
            "subparts": ["torso", "head"]
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
            - Human/Animal: face, torso, entire leg, etc.
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

def get_entity_subparts(client, entity_name, money_manager=None):
    system_prompt = """You are a system that decomposes real-world or conceptual entities into their primary structural or functional subparts. Given the name of an entity, return a JSON object that includes the entity and a list of its subparts.

    ### Instructions:
    - Only include meaningful, commonly recognized subparts.
    - Do not include properties or attributes (e.g., "color", "size").
    - If the entity is too abstract or doesn't have physical parts, return an empty array.

    ### Input:
    Entity: "person"

    ### Output:
    {
    "entity": "person",
    "subparts": ["head", "torso", "arm", "hand", "leg", "foot"]
    }

    ### Input:
    Entity: "car"

    ### Output:
    {
    "entity": "car",
    "subparts": ["engine", "chassis", "wheel", "door", "windshield", "seat"]
    }

    ### Input:
    Entity: \"""" + entity_name + """\"

    ### Output:"""

    response = client.chat.completions.create(
        model="gpt-4o",  # use "gpt-3.5-turbo" if desired
        messages=[
            {"role": "system", "content": system_prompt}
        ],
        temperature=0.2
    )

    # Track costs with money manager
    if money_manager:
        money_manager(response)

    result_text = response.choices[0].message.content.strip()

    try:
        result_json = json.loads(result_text)
        return result_json
    except json.JSONDecodeError:
        print("Could not parse JSON:")
        print(result_text)
        return None
    
def addition_suggest_offset(client, sampled_instance, class_name, image, money_manager=None):
    # Extract instance information
    mask = sampled_instance['pred_mask'].cpu().numpy()
    # Convert mask to grayscale image and encode to base64 (same datatype as base64_image)
    mask_image = (mask * 255).astype(np.uint8)
    base64_mask = encode_image_to_base64(mask_image)
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    system_prompt = """
    Image artifacts are unintended, implausible, or visually corrupted regions that appear in images generated by diffusion models. These artifacts break the natural semantics or visual coherence of an image — for example, a person with extra fingers, a car with distorted wheels, or an animal missing ears. Artifacts are a major concern in both the evaluation and training of image generation models.

    There are three major types of artifacts:
    1. **Addition**: An object part is duplicated and inserted near the original, resulting in anatomically or physically implausible duplication (e.g., an extra ear or thumb).
    2. **Removal**: A part is deleted and the background is inpainted over it (e.g., missing eye, finger, or mirror).
    3. **Distortion**: The part is present but visually malformed, warped, or broken (e.g., twisted wheels or scrambled faces).

    This task focuses solely on **Addition artifacts**.
    """
    prompt = """
    You are an expert Image Artifact Agent specializing in generating **Addition-type artifacts** for images produced by diffusion models.

    ---

    ## Task Overview

    Your role is to simulate **plausible but anatomically incorrect duplications** of object parts by suggesting a spatial offset for duplication.

    You will be given:
    1. An **image** showing an object.
    2. A **segmentation mask** identifying a specific part of the object.
    3. A **textual description** of the entity and the part (e.g., "left ear of a dog").

    Your goal is to determine the most **plausible offset** (x, y) for duplicating this part in a way that resembles real Addition-type artifacts.

    ---

    ## Artifact Guidelines

    - The duplicated part should be placed **adjacent** to the original part in one of four directions: **up, down, left, or right**.
    - The placement should appear **visually coherent** but **anatomically incorrect** (e.g., an extra finger next to other fingers).
    - This is especially common for **terminal or peripheral parts**, such as:
    - **Humans/Animals**: fingers, hands, toes, ears, eyes.
    - **Vehicles**: mirrors, wheels, wipers.

    ---

    ## Offset Placement Constraints

    1. The **target region should not significantly overlap** with the original segmentation mask.
    2. The **target region should minimize overlap with the overall entity mask**.
    3. If the image contains **other instances of the same part** (e.g., other legs or ears), the target should avoid overlapping with them to prevent visual blending.
    4. The **offset should remain reasonably close to the entity** — if too far, diffusion models fail to inject artifacts effectively.
    5. For **peripheral or terminal parts** (e.g., hand, ear), placing the duplicate near the **edge of the entity** is often most effective.
    6. If the duplicated part **blends seamlessly into an existing similar part**, the artifact is not visually recognizable as an addition — avoid this.

    ---

    ## Decision-Making Process

    When choosing the offset, consider:
    - The **semantics** of the object part (from the textual description).
    - The **visual layout and shape** of the segmentation map and image.
    - Known artifact patterns from diffusion model behavior.

    ---

    ## Output Format

    Return your output as a JSON object:

    ```json
    {
    "offset_x": <integer>,
    "offset_y": <integer>
    }
    ```

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
                            "text": system_prompt + prompt + "\n\n" + class_name
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

def addition_sugget_direction(client, sampled_instance, class_name, image, money_manager=None):

    # Extract instance information
    mask = sampled_instance['pred_mask'].cpu().numpy()
    # Convert mask to grayscale image and encode to base64 (same datatype as base64_image)
    mask_image = (mask * 255).astype(np.uint8)
    base64_mask = encode_image_to_base64(mask_image)
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    prompt = """
        You are an expert Image Artifact Agent specializing in Addition-type artifacts for diffusion-generated images. Your role is to simulate plausible image corruption by suggesting where a given part of an object should be duplicated and inserted in the image. Image artifacts are unintended, unrealistic, or semantically incoherent regions in images generated by diffusion models. Among the three main categories—Addition, Removal, and Distortion—this task focuses exclusively on Addition artifacts.

        Addition artifacts occur when a part of an object is implausibly duplicated and inserted adjacent to the original, violating anatomical or physical realism (e.g., a person with an extra thumb, a car with two left mirrors).

        Common characteristics:
            •	Duplication placement: One of four directions — up, down, left, or right — adjacent to the original part.
            •	High likelihood locations: Peripheral or terminal regions (e.g., fingers, ears, mirrors).
            •	Examples:
                •	Human/Animal: fingers, hands, toes, ears, tails.
                •	Vehicles: mirrors, wheels, wipers.

        Task

        Given the following inputs:
            1.	Image (showing an object)
            2.	Segmentation map (highlighting a specific part of the object)
            3.	Textual description of the segmented part (e.g., "left ear of a dog")

        Your goal is to predict the most plausible direction (up, down, left, or right) in which the segmented part could be artificially duplicated, mimicking a realistic yet incorrect addition artifact. Your choice should align with common patterns observed in addition-type artifacts (e.g., extra fingers typically appear beside existing ones).

        Make your decision based on:
            •	The semantics of the object part (from text)
            •	The visual context (from the image and segmentation map)
            •	Known patterns in image artifact generation

        The output format should in json as follows

        {
        "direction": "up" | "down" | "left" | "right"
        }

        ### Input
        \"""" + class_name + """\"
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
    

def visualize_all_candidates(candidate_target_list, image, class_name, output_dir=None, img_filename=None):
    """
    Create a visualization showing all candidate target regions overlaid on the original image
    
    Args:
        candidate_target_list: List of candidate dictionaries with 'target_mask' key
        image: Original image array (numpy array)
        class_name: String describing the part being added
        output_dir: Directory to save the visualization (optional)
        img_filename: Filename for saving (optional)
        
    Returns:
        Visualization image as numpy array
    """
    if not candidate_target_list:
        print("No candidates provided for visualization")
        return image.copy()
    
    # Create a copy of the original image for visualization
    vis_image = image.copy()
    
    # Define colors for different candidates (RGB format)
    candidate_colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green  
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
    ]
    
    print(f"Visualizing {len(candidate_target_list)} candidates for {class_name}")
    
    # Track label positions for text overlay
    label_positions = []
    
    # Overlay each candidate mask with a different color
    for i, candidate in enumerate(candidate_target_list):
        if i >= len(candidate_colors):
            break  # Limit to available colors
            
        target_mask = candidate['target_mask']
        color = candidate_colors[i]
        
        # Find mask pixels and apply color overlay
        mask_indices = np.where(target_mask > 0)
        if len(mask_indices[0]) > 0:
            # Apply semi-transparent overlay
            alpha = 0.6
            vis_image[mask_indices[0], mask_indices[1]] = (
                alpha * np.array(color) + (1 - alpha) * vis_image[mask_indices[0], mask_indices[1]]
            ).astype(np.uint8)
            
            # Find center of mass for label placement
            center_y = int(np.mean(mask_indices[0]))
            center_x = int(np.mean(mask_indices[1]))
            label_positions.append((center_x, center_y, i))
            
            # Print candidate information
            print(f"  Candidate {i}: entity_overlap={candidate.get('entity_overlap', 'N/A'):.3f}, "
                  f"inter_overlap={candidate.get('inter_overlap', 'N/A'):.3f}, "
                  f"intra_overlap={candidate.get('intra_overlap', 'N/A'):.3f}")
    
    # Add text labels using matplotlib if available
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(vis_image)
    ax.set_title(f'Candidate Addition Regions for {class_name}', fontsize=14, fontweight='bold')
        
    # Add numbered labels at candidate centers
    for center_x, center_y, idx in label_positions:
        ax.text(center_x, center_y, str(idx), fontsize=16, fontweight='bold', 
               ha='center', va='center', color='white',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.8))
        
    # Create legend
    legend_elements = []
    for i in range(min(len(candidate_target_list), len(candidate_colors))):
        color_norm = [c/255.0 for c in candidate_colors[i]]  # Normalize to [0,1]
        legend_elements.append(patches.Patch(color=color_norm, label=f'Candidate {i}'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.axis('off')
        
    # Save the enhanced visualization
    if output_dir and img_filename:
        img_base = os.path.splitext(img_filename)[0]
        img_dir = os.path.join(output_dir, img_base)
        os.makedirs(img_dir, exist_ok=True)
            
        vis_path = os.path.join(img_dir, "00_candidate_visualization.png")
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Enhanced candidate visualization saved to: {vis_path}")
            
        # Also save the simple overlay version
        simple_vis_path = os.path.join(img_dir, "00_candidate_overlay.png")
        cv2.imwrite(simple_vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    
    # Save visualization if output path is provided (for both cases)
    if output_dir and img_filename:
        img_base = os.path.splitext(img_filename)[0]
        img_dir = os.path.join(output_dir, img_base)
        os.makedirs(img_dir, exist_ok=True)
        
        # Save the visualization
        vis_path = os.path.join(img_dir, "00_candidate_visualization.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Candidate visualization saved to: {vis_path}")
    
    return vis_image

def visualize_candidate_images_for_api(candidate_images, class_name, output_dir=None, img_filename=None):
    """
    Create a visualization showing all candidate images that will be sent to OpenAI API
    
    Args:
        candidate_images: List of base64 encoded candidate images
        class_name: String describing the part being added
        output_dir: Directory to save the visualization (optional)
        img_filename: Filename for saving (optional)
    """
    if not candidate_images:
        print("No candidate images to visualize")
        return
    
    # Convert base64 images back to numpy arrays
    candidate_arrays = []
    for base64_img in candidate_images:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_img)
        # Convert to PIL Image
        pil_img = Image.open(io.BytesIO(img_bytes))
        # Convert to numpy array
        img_array = np.array(pil_img)
        candidate_arrays.append(img_array)
    
    # Create a grid visualization
    n_candidates = len(candidate_arrays)
        
    if n_candidates <= 2:
        rows, cols = 1, n_candidates
        figsize = (6 * cols, 6)
    elif n_candidates <= 4:
        rows, cols = 2, 2
        figsize = (12, 12)
    else:
        rows = (n_candidates + 3) // 4  # Round up division
        cols = 4
        figsize = (16, 4 * rows)
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
        
    for i, img_array in enumerate(candidate_arrays):
        axes[i].imshow(img_array)
        axes[i].set_title(f'Candidate {i}\n(Sent to OpenAI API)', fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
    # Hide unused subplots
    for i in range(len(candidate_arrays), len(axes)):
        axes[i].axis('off')
        
    plt.suptitle(f'OpenAI API Input: Candidate Images for {class_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
        
    # Save the visualization
    if output_dir and img_filename:
        img_base = os.path.splitext(img_filename)[0]
        img_dir = os.path.join(output_dir, img_base)
        os.makedirs(img_dir, exist_ok=True)
            
        api_vis_path = os.path.join(img_dir, "00_openai_api_candidates.png")
        plt.savefig(api_vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"OpenAI API candidate images visualization saved to: {api_vis_path}")
    else:
        plt.show()
    

def addition_select_candidate(client, candidate_target_list, class_name, image, output_dir=None, img_filename=None, money_manager=None):
    """
    Select the best candidate mask for addition artifacts using OpenAI Vision API
    
    Args:
        client: OpenAI client
        candidate_target_list: List of candidate dictionaries, each containing:
            - 'target_mask': numpy array of the candidate mask
            - 'target_bbox': bounding box coordinates
            - 'offset': offset values
            - 'entity_overlap', 'inter_overlap', 'intra_overlap': overlap metrics
            - 'radius', 'angle': positioning parameters
        class_name: String describing the part being added (e.g., "hand of person")
        image: Image array (numpy array)
        output_dir: Directory to save visualizations (optional)
        img_filename: Filename for saving visualizations (optional)
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        Best candidate dictionary or None if selection fails
    """
    if not candidate_target_list:
        print("No candidates provided for selection")
        return None
    
    if len(candidate_target_list) == 1:
        print("Only one candidate provided, returning it")
        return candidate_target_list[0]
    
    # Encode original image to base64
    base64_image = encode_image_to_base64(image)
    
    # Create visualization showing all candidates
    candidate_images = []
    for i, candidate in enumerate(candidate_target_list):
        # Create a copy of the original image
        img_with_mask = image.copy()
        target_mask = candidate['target_mask']
        
        # Overlay the candidate mask in a distinct color (e.g., red)
        mask_indices = np.where(target_mask > 0)
        if len(mask_indices[0]) > 0:
            img_with_mask[mask_indices[0], mask_indices[1]] = [255, 0, 0]  # Red overlay
        
        # Convert to base64
        base64_candidate = encode_image_to_base64(img_with_mask)
        candidate_images.append(base64_candidate)

    # Visualize the candidate images that will be sent to OpenAI API
    visualize_candidate_images_for_api(candidate_images, class_name, 
                                     output_dir=output_dir, img_filename=img_filename)
    
    # Create the prompt for candidate selection
    prompt = f"""
    You are an expert Image Artifact Agent specializing in Addition-type artifacts for diffusion-generated images. 
    Your task is to select the most plausible candidate location for duplicating a "{class_name}" in the image.

    You will see the original image followed by {len(candidate_target_list)} candidate positions, each showing where the "{class_name}" could be duplicated (highlighted in red).

    Addition artifacts should look realistic but anatomically/structurally incorrect. Consider:

    1. **Visual Plausibility**: The duplicate should look like it could naturally belong in that location
    2. **Anatomical Context**: For body parts, consider natural positioning and proportions
    3. **Spatial Relationships**: The duplicate should maintain realistic spatial relationships with the original
    4. **Lighting & Perspective**: The location should match the lighting and perspective of the original
    5. **Structural Logic**: For objects, consider how additional parts would logically be positioned

    Select the candidate that would create the most believable yet incorrect duplication.

    Return your selection in this exact JSON format:
    {{
        "selected_candidate": <candidate_number>,
        "reasoning": "<brief explanation of why this candidate is best>"
    }}

    Where candidate_number is 0, 1, 2, or 3 corresponding to the order shown.
    """
    
    # Prepare messages for API call
    messages_content = [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
    
    # Add candidate images
    for i, candidate_image in enumerate(candidate_images):
        messages_content.append({
            "type": "text",
            "text": f"Candidate {i}:"
        })
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{candidate_image}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        # Track costs with money manager
        if money_manager:
            money_manager(response)
        
        response_text = response.choices[0].message.content.strip()
        print(response_text)
        # Try to extract JSON from the response
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            result = json.loads(json_match.group())
            selected_idx = result['selected_candidate']
            reasoning = result['reasoning']
            
            # Validate selection index
            if 0 <= selected_idx < len(candidate_target_list):
                print(f"Selected candidate {selected_idx}: {reasoning}")
                return candidate_target_list[selected_idx]
            else:
                print(f"Invalid candidate index {selected_idx}, using first candidate")
                return candidate_target_list[0]
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract candidate number from text
            print(f"Could not parse selection from response: {response_text}")
            print("Using first candidate as fallback")
            return candidate_target_list[0]
    except Exception as e:
        print(f"Error in candidate selection: {e}")
        print("Using first candidate as fallback")
        return candidate_target_list[0]
    

def artifact_type_decision(client, sampled_instance, image, money_manager=None):
    """
    Analyze a sampled instance using OpenAI Vision API
    
    Args:
        sampled_instance: The sampled instance object (contains bbox, scores, etc.)
        class_name: The name of the detected class
        image: The image array (numpy array)
        client: OpenAI client (if None, uses default openai module)
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        The response from OpenAI API or None if error
    """

    # Extract instance information
    mask = sampled_instance.pred_masks[0].cpu().numpy()
    # Convert mask to grayscale image and encode to base64 (same datatype as base64_image)
    mask_image = (mask * 255).astype(np.uint8)
    base64_mask = encode_image_to_base64(mask_image)
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    prompt = """
    1 · Purpose

    You will decide which kind of visual artifact to inject into a chosen part of an image so that downstream models can learn to detect or fix similar errors in real photos.

    ⸻

    2 · Inputs you receive
        1.	Scene description – plain-language explanation of the image.
        2.	Segmentation map – pixel- or patch-level masks for a single part.
    ⸻

    3 · Artifact operations you may choose
        •	Addition
    Copy the selected part and paste the copy just outside the original mask.
    When to choose: only if the part is small or medium – e.g. fingers, ears, leaves. Avoid duplicating very large objects.
        •	Removal
    Erase the selected part and in-paint the background to fill the hole.
    When to choose: only if the part is small, so the gap can be concealed. Do not remove large regions.
        •	Distortion
    Keep the part but warp or twist its pixels so the shape is obviously wrong.
    When to choose: best for large, continuous regions such as faces, torsos, wheels.

    ⸻

    4 · Extra rule for Addition

    If you pick Addition you must also state where the copy will appear relative to the original mask: up, down, left, or right.

    ⸻

    5 · Decision process (internal)
        1.	Check the mask size.
        2.	Match the size and context against the guidelines in §3.
        3.	Select Addition, Removal, or Distortion.
        4.	If Addition, determine the best cardinal direction for the copy.
        5.	Do not reveal these thoughts; output only the schema below.


    6 · Exact output you must return
    {
    "artifact_type": "addition" | "removal" | "distortion",
    "direction": "up" | "down" | "left" | "right" | null
    }


    Set "direction" to null unless "artifact_type" is "addition".
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
        