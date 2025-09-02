import json
from PIL import Image
import base64
import io
import numpy as np
import re
from openai.types.chat import ChatCompletion
from typing import Union, List, Optional
from pydantic import BaseModel

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

IN_CONTEXT_EXAMPLES = {
    "addition": {
        "positive": {
            "explanation": " The zebra has an extra ear on the middle of its head.",
            "object_name": "an ear of zebra",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/positive/artifact_target.png'))
        },
        "negative": {
            "explanation": "This is Case-A; the original image already had an ear of a cat in the image, while the artifact image did not duplicate it.",
            "object_name": "an ear of a cat",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/negative/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/negative/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/negative/artifact_target.png'))
        }
    },
    "removal": {
        "positive": {
            "explanation": "The bird is missing its tail, where the bird should have a tail.",
            "object_name": "a tail of a bird",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/removal/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/removal/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/removal/positive/artifact_target.png'))
        },
        "negative": {
            "explanation": "Although the ear of a dog is manipulated, it did not remove the ear of the dog.",
            "object_name": "an ear of a dog",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/removal/negative/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/removal/negative/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/removal/negative/artifact_target.png'))
        }
    },
    "distortion": {
        "positive": {
            "explanation": "The face of the person is distorted, as the facial features are not defined well.",
            "object_name": "a hand of a person",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/distortion/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/distortion/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/distortion/positive/artifact_target.png'))
        }
    },
    "fusion": {
        "positive": {
            "explanation": "The heads of two cats are merged into one.",
            "object_name": "a cat and a cat",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/positive/artifact_target.png'))
        },
        "negative": {
            "explanation": "Although the boundary of the two elephant has been manipulated, the boundary between the two elephants is still visible in the artifact image.",
            "object_name": "an elephant and a baby elephant",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/negative/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/negative/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/negative/artifact_target.png'))
        }
    }
}

class EntitySubentityResponse(BaseModel):
    entity: str
    subentities: List[str]

class VocabResponse(BaseModel):
    peripheral: Optional[List[EntitySubentityResponse]] = None
    intermediate: Optional[List[EntitySubentityResponse]] = None

class ArtifactSuccessResponse(BaseModel):
    reasoning: str
    success: bool   

class ArtifactExplanationResponse(BaseModel):
    explanation: str

class ArtifactDescriptionResponse(BaseModel):
    has_artifact: bool
    explanation: str
    
def get_entity_subentities(client, image, money_manager=None):
    base64_image = encode_image_to_base64(image)
    system_prompt = """
    You are given an image. Identify the visible **entities** and their **subentities**, split into two layers:
    - **Peripheral (outermost) subentities** first (e.g., hands, fingers, eyes, ears, paws, tails, wheels, mirrors).
    - **Intermediate/Core subentities** second (e.g., arms, legs, face, head, door, window).

    Output must return two dictionaries for each peripheral and intermediate types:

    peripheral: {
        "<peripheral_entity1>": ["<peripheral_sub1>", "<peripheral_sub2>", "..."],
        "<peripheral_entity2>": ["<peripheral_sub1>", "<peripheral_sub2>", "..."],
        ...
    },
    intermediate: {
        "<intermediate_entity1>": ["<intermediate_sub1>", "<intermediate_sub2>", "..."],
        "<intermediate_entity2>": ["<intermediate_sub1>", "<intermediate_sub2>", "..."],
        ...
    }

    Hard rules:
    1) Each returned entity MUST have **at least one** subentity in its dictionary. **Never** return an empty list for any entity.
    2) If you cannot name at least one clearly visible subentity for an entity **in that layer**, omit that entity from that dictionary.
    3) Subentities must be **clearly visible** and **reasonably segmentable** in the image.
    4) **Exclude** parts that are tightly bound to or visually fused with the torso/core body (e.g., arms pressed to sides, folded wings against body). Only include subentities with clear visual separation.
    5) Do **not** invent parts that are occluded, cropped out, or ambiguous.
    6) Use concise, lowercase **nouns**; de-duplicate terms. Prefer 1–6 subentities per entity per dictionary.
    7) **Return exactly one JSON object** with exactly two top-level keys: **"peripheral"** and **"intermediate"** (not an array, not multiple objects).
    8) **Granularity rule (coarsity):** Choose the **most specific visible entity**. If only a part is clearly visible (e.g., a leg without enough evidence of the full person), output that part as the entity (e.g., "leg") rather than its parent (e.g., "person"). Do **not** infer parent entities that are not clearly visible.
    9) **Peripheral variable-cardinality ban:** In the **peripheral** dictionary, **do not** include variable-cardinality micro-parts (0..n multiplicity) such as leaves, hairs, feathers, scales, spikes, thorns, grains, pebbles, raindrops, confetti, crowd members, etc. Prefer distal parts with fixed or small bounded counts (e.g., hand, finger, ear, eye, wheel, mirror). If only variable-cardinality micro-parts are visible for an entity, **omit that entity** from the peripheral dictionary.

    Layering guidance:
    - **Peripheral dictionary (index 0):** Include outermost parts with clear boundaries (e.g., arm, leg, wing, fin, hand, finger, nail, ear, eye, paw, tail, wheel, mirror, antenna).
    - For the **peripheral** dictionary, exclude variable-cardinality micro-parts (e.g., leaves, hairs, feathers, scales, spots/pattern dots, raindrops); select fixed-cardinality or bounded-count distal parts instead.
    - **Intermediate/Core dictionary (index 1):** Include mid-level structural parts (e.g., arm, leg, face, door, window).
    - If a subentity fits both notions, prefer **peripheral** only if it is clearly distal and separable (e.g., "hand" → peripheral; "arm" → intermediate).
    - If a candidate layer has no valid entities with visible subentities, set that key to an empty object `[]` (e.g., `"peripheral": []` or `"intermediate": []`).

    Clarifications:
    - Examples of valid subentities:
    • person → face, arm, hand, leg, foot, ear, eye
    • hand → finger, nail, palm
    • dog → ear, snout, leg, tail, paw
    • car → wheel, door, window, mirror
    - Avoid generic torso-like regions. If no fine-grained parts are clearly separable for a candidate entity, **omit the entity** rather than returning an empty list.
    - Granularity examples:
    • If only a single **leg** is clearly visible:
    {
        "peripheral": {"leg": ["knee", "ankle", "foot"]},
        "intermediate": []
    }
    - Variable-cardinality micro-parts are **not allowed** in the peripheral dictionary. Examples to avoid: leaf/leaves (tree), hair/hairs (person/animal), feather/feathers (bird), scale/scales (fish/reptile), spot/spots (dalmatian), petal/petals (flower), book/books (bookshelf), crowd/persons (crowd scene).

    Bad examples (NOT allowed):
    - Returning a single array or more than one top-level JSON object
    - {"dog": []}  # empty subentities list
    - Multiple separate top-level JSON objects
    - "peripheral": [{"entity": "tree", "subentities": ["leaf", "fruit"]}]  # variable-cardinality micro-parts in peripheral

    Good format examples (illustrative only):
    {
        "peripheral": [{"entity": "person", "subentities": ["hand", "finger", "leg", "arm"]}, {"entity": "car", "subentities": ["wheel", "mirror"]}, {"entity": "dog", "subentities": ["ear", "paw", "leg", "tail"]}],
        "intermediate": [{"entity": "person", "subentities": ["face", "palm"]}, {"entity": "car", "subentities": ["door", "window"]}, {"entity": "dog", "subentities": ["leg", "face"]}]
    }

    {
        "peripheral": [{"entity": "cat", "subentities": ["ear", "paw", "leg", "tail"]}, {"entity": "bicycle", "subentities": ["wheel", "pedal"]}],
        "intermediate": [{"entity": "cat", "subentities": ["face"]}, {"entity": "bicycle", "subentities": ["frame", "seat"]}]
    }

    Edge-case examples:
    - Only distal parts visible:
    {
        "peripheral": [{"entity": "hand", "subentities": ["finger", "nail"]}],
        "intermediate": []
    }

    - Only intermediate parts visible:
    {
        "peripheral": [],
        "intermediate": [{"entity": "person", "subentities": ["face", "palm"]}]
    }

    - Multiple distal entities visible, no intermediate:
    {
        "peripheral": [{"entity": "hand", "subentities": ["finger", "nail"]}, {"entity": "dog", "subentities": ["ear", "paw", "tail", "leg"]}],
        "intermediate": []
    }
    """

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ],
            temperature=0.2,
            text_format=VocabResponse
        )

        if money_manager:
            money_manager(response)

        return response.output_parsed

    except Exception as e:
        print(f"Error analyzing sampled instance: {e}")
        return None
    
def artifact_description(client, masked_original_image, target_original_image, target_artifact_image, object_name, artifact_type, money_manager=None):
    """
    Combined function that filters artifacts and generates explanations in a single API call.
    
    Combines both detection logic and explanation generation into one prompt for efficiency.
    
    Args:
        client: OpenAI client
        masked_original_image: Original image with target region masked out
        target_original_image: Original image showing only the target region  
        target_artifact_image: Artifact image showing only the target region
        object_name: Description of the object (e.g., "a hand of a person")
        artifact_type: Either "addition", "removal", "fusion", or "distortion"
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        ArtifactDescriptionResponse: Contains 'has_artifact' boolean, 'reasoning' string for detection, 
                                   and 'explanation' string describing the artifact
    """
    # Encode images to base64
    original_masked = encode_image_to_base64(masked_original_image)
    original_target = encode_image_to_base64(target_original_image)
    artifact_target = encode_image_to_base64(target_artifact_image)

    # Combined instruction for both detection and explanation
    combined_instruction = {
        "addition": (
            "You are an expert at detecting and describing addition-type artifacts in AI-generated images.\n\n"
            "Addition artifacts occur when a part of an object is duplicated and placed adjacent to the original, "
            "creating anatomically or structurally implausible duplications (e.g., extra fingers, duplicate ears, duplicate wheels).\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The object name that may be added\n\n"
            "Your task is to:\n"
            "1. **DETECT**: Determine if there is a successful addition artifact in the third image\n"
            "2. **EXPLAIN**: If an artifact is present, describe what looks wrong in the target region\n\n"
            "Detection criteria for addition artifacts:\n"
            "\t• **Case A — Object present in target (Image 2)**: If the second image already contains the specified object, then the third image must show a **plausible duplication** or **additional instance** of the object. The new instance should be **distinct** from the original.\n"
            "\t• **Case B — No object in target (Image 2)**: If the second image does **not** contain the object, the third image must show a **clearly new instance** with a **distinct boundary/contour**.\n\n"
            "Reject if there is only subtle texture change, brightness shift, or local warping rather than a new instance.\n\n"
            "For explanation: Focus on cues like duplicated parts, unnatural growths, or extra elements that conflict with normal anatomy or structure."
        ),
        "removal": (
            "You are an expert at detecting and describing removal-type artifacts in AI-generated images.\n\n"
            "Removal artifacts occur when a part of an object is deleted and the area is inpainted with background, "
            "resulting in missing features or gaps where something should be present (e.g., missing fingers, absent ears).\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The object name that may be removed\n\n"
            "Your task is to:\n"
            "1. **DETECT**: Determine if there is a successful removal artifact in the third image\n"
            "2. **EXPLAIN**: If an artifact is present, describe what looks wrong in the target region\n\n"
            "Detection criteria (be strict):\n"
            "\t• **Definitive removal evidence**: stump/termination cues, disrupted silhouette, hollow/negative space, texture continuation/inpainting traces, mismatched shadows/reflections, or symmetry break.\n"
            "\t• **Ambiguity/occlusion rule**: If the missing part could plausibly be merely *hidden*, set has_artifact = false.\n"
            "\t• **Anatomical plausibility check**: If the scene still reads as anatomically correct and a typical pose could hide the part, set has_artifact = false.\n\n"
            "For explanation: Focus on cues like missing structure, unnatural gaps, smoothed-over areas, or anatomical discontinuity where something appears to be absent."
        ),
        "fusion": (
            "You are an expert at detecting and describing fusion-type artifacts in AI-generated images.\n\n"
            "Fusion artifacts occur when parts or distinct entities are unnaturally merged together, "
            "creating blurred boundaries, overlapped textures, or structural entanglement (e.g., two animals merged into one).\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The two object names that may be fused\n\n"
            "Your task is to:\n"
            "1. **DETECT**: Determine if there is a successful fusion artifact in the third image\n"
            "2. **EXPLAIN**: If an artifact is present, describe what looks wrong in the target region\n\n"
            "Detection criteria (be strict):\n"
            "\t• **Boundary visibility comparison**: Compare Image 2 and Image 3. If a clear, continuous boundary between the two objects remains visible in Image 3 (similar to Image 2), set has_artifact = false.\n"
            "\t• **Fusion cues needed**: boundary loss/softening across seams; cross-object texture/color blending; geometry interpenetration; inconsistent occlusion ordering.\n"
            "\t• **Not fusion**: mere blur, minor warping, or lighting change that preserves recognizable boundary.\n\n"
            "For explanation: Focus on cues like boundary loss/softening across seams, cross-object texture/color bleed, geometry interpenetration, and inconsistent occlusion ordering."
        ),
        "distortion": (
            "You are an expert at describing distortion-type artifacts in AI-generated images.\n\n"
            "Distortion artifacts occur when parts are warped, creating unnatural geometry, irregular textures, or visual blending errors.\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The object name that may be distorted\n\n"
            "Your task is to:\n"
            "1. **EXPLAIN**: Describe what looks wrong or unnatural in the target region\n\n"
            "For explanation: Focus on cues like warped shapes, unnatural geometry, irregular textures, or visual blending errors that make the structure appear broken or malformed."
        )
    }

    prompt = f"""
    {combined_instruction[artifact_type]}
    
    Return your analysis in the following format:
    "has_artifact": true/false (whether the artifact is successfully present, always true for distortion)
    "explanation": "Detailed description of what looks wrong in the region (empty string if no artifact)"
    
    Important: 
    - Focus on visible evidence in the target region
    - Use simple, clear language for explanations
    - Do not refer to images by number; say "the region" or "the area" instead
    - Make explanations understandable for non-experts
    """

    try:
        # Handle distortion separately since it has no in-context examples
        if artifact_type == "distortion":
            response = client.responses.parse(
                # Add a positive in-context learning example for distortion
                model="gpt-4o",
                input=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['original_masked']}"},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['original_target']}"},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['artifact_target']}"},
                            {"type": "input_text", "text": f"{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['object_name']}"}
                            ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f'{{"has_artifact": true, "explanation": "{IN_CONTEXT_EXAMPLES["distortion"]["positive"]["explanation"]}"}}'}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_masked}"},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_target}"},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{artifact_target}"},
                            {"type": "input_text", "text": f"{object_name}"}
                        ]
                    }
                ],
                temperature=0.2,
                text_format=ArtifactDescriptionResponse
            )
        else:
            # Use in-context examples for addition, removal, fusion
            response = client.responses.parse(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['original_masked']}"},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['original_target']}"},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['artifact_target']}"},
                                    {"type": "input_text", "text": f"{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['object_name']}"}
                                ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f'{{"has_artifact": true, "explanation": "{IN_CONTEXT_EXAMPLES[artifact_type]["positive"]["explanation"]}"}}'}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['original_masked']}"},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['original_target']}"},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['artifact_target']}"},
                                    {"type": "input_text", "text": f"{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['object_name']}"}
                                ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f'{{"has_artifact": false, "explanation": ""}}'}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_masked}"},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_target}"},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{artifact_target}"},
                                    {"type": "input_text", "text": f"{object_name}"}
                                ]
                    }
                ],
                temperature=0.2,
                text_format=ArtifactDescriptionResponse
            )

        # Track costs with money manager
        if money_manager:
            money_manager(response)

        return response.output_parsed

    except Exception as e:
        print(f"Error in artifact description ({artifact_type}): {e}")
        return ArtifactDescriptionResponse(
            has_artifact=False,
            explanation=""
        )

def artifact_success(client, masked_original_image, target_original_image, target_artifact_image, object_name, artifact_type, money_manager=None):
    """
    Unified query to check if an artifact injection (addition, removal, or fusion) was successful.

    Args:
        client: OpenAI client
        original_image: Original image as numpy array
        artifact_image: Modified image with artifact as numpy array
        object_name: Description of the object (e.g., "a hand of a person")
        artifact_type: Either "addition", "removal", or "fusion"
        money_manager: MoneyManager instance for cost tracking (optional)

    Returns:
        dict: Contains 'success' boolean, and 'reasoning' string. On parse errors, includes 'raw_response'.
    """
    # Encode images to base64
    original_masked = encode_image_to_base64(masked_original_image) # mask out the target region
    original_target = encode_image_to_base64(target_original_image) # show only the target region
    artifact_target = encode_image_to_base64(target_artifact_image) # show only the target region

    instruction  = {
        "addition": (
            "You are an expert at detecting addition-type artifacts in AI-generated images.\n\n"
            "Addition artifacts occur when a part of an object is duplicated and placed adjacent to the original, "
            "creating anatomically or structurally implausible duplications (e.g., extra fingers, duplicate ears, duplicate wheels). "
            "Your role is to determine if the artifact was successfully injected into the original image on the target region.\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The object name that will be added\n"
            f'Your task is to determine if there is the object present in the third image, in the target region.\n\n'
            "Addition artifact success is determined as follows:\n"
            "\t• **Case A — Object present in target (Image 2)**: If the second image (target region of the original) already contains the specified object, then the third image must show a **plausible duplication** or **additional instance** of the object in the target region. The new instance should be **distinct** from the original and not just a local warping or minor change. Duplication should be anatomically/structurally plausible for the object.\n"
            "\t• **Case B — No object in target (Image 2)**: If the second image does **not** contain the object in the target region, focus on the **third image** only. Success requires that the third image shows a **clearly new instance** of the specified object **within the target region**, with a **distinct boundary/contour** (not a texture smear or brightness change) and appearance consistent with that object. Do **not** require proximity or adjacency to any other instance in the scene.\n"
            "\n"
            "Checklist for addition artifact success (all must pass):\n"
            "\t1) The third image (artifact region) contains the specified object in the target region.\n"
            "\t2) The object's appearance is consistent with the class/type (e.g., an ear looks like an ear).\n"
            "\t3) If Image 2 lacked the object in target, Image 3 shows a **clearly new** instance inside the target region with a **distinct boundary** (not just local warping/smearing).\n"
            "\n"
            "Reject if:\n"
            "\t• There is only a subtle texture change, brightness shift, or local warping rather than a new instance.\n"
            "\t• The added object is not visually separable or lacks clear boundaries.\n"
        ),
        "removal": (
            "You are an expert at detecting removal-type artifacts in AI-generated images.\n\n"
            "Removal artifacts occur when a part of an object is deleted and the area is inpainted with background, "
            "resulting in missing features or gaps where something should be present (e.g., missing fingers, absent ears). "
            "Your role is to determine if the artifact was successfully injected into the original image on the target region, "
            "AND to reject cases where the scene remains anatomically/plausibly correct due to occlusion or viewpoint.\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The object name that will be removed\n"
            f"Your task is to determine if the object is truly absent in the third image, in the target region, and that its absence is *not* plausibly explained by occlusion, pose, or viewpoint.\n\n"
            "Evaluate using the following rules (be strict):\n"
            "\t• **Definitive removal evidence** (at least one should be visible): stump/termination cues, disrupted silhouette, hollow/negative space where the part should be, texture continuation/inpainting traces across the expected attachment point, mismatched shadows/reflections, or symmetry break that cannot be explained by pose.\n"
            "\t• **Ambiguity/occlusion rule (hard filter)**: If the missing part could plausibly be merely *hidden* (e.g., a cat's tail could be behind the torso, a mug handle could be removed, but can still be seen as a mug), set success = false.\n"
            "\t• **Anatomical plausibility check**: If the scene still reads as anatomically correct (no clear gap, no silhouette disruption, no attachment artifact) *and* a typical pose could hide the part, set success = false.\n"
            "\t• **Context consistency**: If shadows, reflections, or contact points imply the part should be visible but isn't, that supports success = true.\n\n"
            "Quick checklist (all must pass for success = true):\n"
            "\t1) Clear visual absence in the target area (not just low contrast).\n"
            "\t2) At least one removal cue (stump/gap/inpainting/silhouette break/shadow mismatch).\n"
            "\t3) No plausible occlusion or viewpoint explanation.\n\n"
        ),
        "fusion": (
            "You are an expert at detecting fusion-type artifacts in AI-generated images.\n\n"
            "Fusion artifacts occur when a part or two distinct entities are unnaturally merged together, "
            "creating blurred boundaries, overlapped textures, or structural entanglement that makes the separation implausible "
            "(e.g., two animals merged into one, a limb merged into the torso, overlapping facial features). "
            "Your role is to determine if the artifact was successfully injected into the original image on the target region.\n\n"
            "You will be shown:\n"
            "\t1. An original image without the target region\n"
            "\t2. An original image with only the target region\n"
            "\t3. An artifact image with only the target region\n"
            "\t4. The two object names that will be fused\n"
            f"Your task is to determine if there is an unnatural fusion involving the two objects present in the third image, in the target region.\n\n"
            "Evaluate using the following rules (be strict):\n"
            "\t• **Boundary visibility comparison — HARD FILTER**: Compare Image 2 (original target) and Image 3 (artifact target). If a clear, continuous boundary/contour between the two objects remains visible in Image 3 (similar to Image 2), set success = false. Visible separation implies no fusion.\n"
            "\t• **Fusion cues (need at least one)**: boundary loss or severe softening across the seam; cross-object texture/color blending; geometry interpenetration or topological entanglement; inconsistent occlusion ordering (parts that should be in front/behind become ambiguous).\n"
            "\t• **Not fusion**: mere blur, minor warping, or lighting change that preserves a recognizable boundary line between objects.\n\n"
            "Quick checklist (all must pass for success = true):\n"
            "\t1) Boundary between the two objects in Image 3 is degraded or missing relative to Image 2 (no clean contour separation).\n"
            "\t2) At least one strong fusion cue is present (texture bleed, geometry merge, occlusion inconsistency).\n"
            "\t3) The region reads as a single merged structure rather than two distinct adjacent objects.\n\n"
            "Output target: Decide success strictly. Favor false if uncertain.\n"
        ),
    }

    prompt = f"""
    {instruction[artifact_type]}
    Return your analysis in the following format:
    "reasoning": "Brief explanation of what you observe in the masked region from the third image"
    "success": true/false,
    """

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['original_masked']}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['original_target']}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['artifact_target']}"},
                                {"type": "input_text", "text": f"{IN_CONTEXT_EXAMPLES[artifact_type]['positive']['object_name']}"}
                            ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": f'{{"reasoning": "{IN_CONTEXT_EXAMPLES[artifact_type]["positive"]["reasoning"]}", "success": true}}'}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['original_masked']}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['original_target']}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['artifact_target']}"},
                                {"type": "input_text", "text": f"{IN_CONTEXT_EXAMPLES[artifact_type]['negative']['object_name']}"}
                            ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": f'{{"reasoning": "{IN_CONTEXT_EXAMPLES[artifact_type]["negative"]["reasoning"]}", "success": false}}'}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_masked}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_target}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{artifact_target}"},
                                {"type": "input_text", "text": f"{object_name}"}
                            ]
                }
            ],
            temperature=0.2,
            text_format=ArtifactSuccessResponse
        )

        # Track costs with money manager
        if money_manager:
            money_manager(response)

        return response.output_parsed

    except Exception as e:
        print(f"Error in artifact query ({artifact_type}): {e}")
        return None


def artifact_explanation_from_triplet(
    client,
    masked_original_image,
    target_original_image,
    target_artifact_image,
    object_name,
    artifact_type,
    money_manager=None,
):
    """
    Generate a natural language explanation of the injected artifact using the *same inputs* as
    `artifact_success`: three images (masked original, original target crop, artifact target crop)
    and one object name, while making the VLM explicitly adhere to the provided artifact type
    ("addition", "removal", or "fusion").

    Inputs:
      - masked_original_image: original image with the target region masked out (numpy or PIL)
      - target_original_image: original image showing only the target region
      - target_artifact_image: artifact image showing only the target region
      - object_name: textual description of the object/objects (e.g., "an ear of a cat", or for fusion: "a sheep and a sheep")
      - artifact_type: one of {"addition","removal","fusion"}

    Returns:
      ArtifactExplanationResponse (pydantic) with a single natural-language `explanation` string.
    """

    # Encode images
    original_masked = encode_image_to_base64(masked_original_image)
    original_target = encode_image_to_base64(target_original_image)
    artifact_target = encode_image_to_base64(target_artifact_image)

    # Type-specific guidance to FORCE adherence to the artifact type semantics
    type_guidance = {
        "addition": (
            "You are describing an **ADDITION** artifact.\n"
            "An addition artifact introduces a *new instance* of the specified object part in the target region.\n"
            "Focus on cues like:duplicated parts, unnatural growths, or extra elements that conflict with normal anatomy or structure.\n"
            "and avoid calling mere local warps or texture changes an addition. If Image 2 already had the part,explain how Image 3 shows a second instance; if Image 2 lacked it, explain how Image 3 introduces a clear, separable instance consistent with the object class."
        ),
        "removal": (
            "You are describing a **REMOVAL** artifact.\n"
            "A removal artifact deletes the specified part, replacing it with inpainted/background content.\n"
            "Focus on cues like: missing structure, unnatural gaps, smoothed-over areas, or anatomical discontinuity where something appears to be absent.\n"
            "If the absence could be plausibly explained by occlusion or viewpoint, acknowledge that ambiguity."
        ),
        "distortion": (
            "You are describing a **DISTORTION** artifact.\n"
            "A distortion artifact warps the specified part, creating unnatural geometry, irregular textures, or visual blending errors.\n"
            "Focus on cues like: warped shapes, unnatural geometry, irregular textures, or visual blending errors that make the structure appear broken or malformed.\n"
        ),
        "fusion": (
            "You are describing a **FUSION** artifact.\n"
            "A fusion artifact unnaturally merges two parts or entities, degrading or erasing the boundary between them.\n"
            "Focus on cues like: boundary loss/softening across seams, cross-object texture/color bleed, geometry interpenetration, and inconsistent occlusion ordering. If a clean boundary remains, note that as evidence against fusion."
        ),
    }

    # Build the instruction prompt
    if artifact_type not in type_guidance:
        raise ValueError(f"Unsupported artifact_type: {artifact_type}")

    prompt = f"""
You will receive three images (in this order) and an object name:
  1) Original image WITHOUT the target region (masked)
  2) Original image showing ONLY the target region
  3) Artifact image showing ONLY the target region (this is the image to describe)
  4) The **object name** regarding the injected artifact

TASK: Write a concise one sentence description describing what looks wrong in **Image 3** consistent with the given artifact type.
Be specific about visible cues. When helpful, you may implicitly contrast with Image 2 to justify your claim (e.g., boundary changes), but DO NOT refer to the images by number; say "the region" or "the area" instead.

{type_guidance[artifact_type]}

Strict rules:
- Focus on visible evidence in the provided region; do not speculate about unseen areas.
- Do not output JSON or code fences. Return plain natural language only.
- Explicitly frame your explanation in terms of the **object name** provided, making clear how the abnormality relates to that object being an artifact.
- Use simple, clear language in your explanation. Avoid technical jargon unless it is necessary to describe the artifact. Make your description easy to understand for someone without expert knowledge.
- Do not justify why this is an artifact injection. Only describe the visible abnormality in the image region, without reasoning about generation or intent.
- Do not describe the artifact type, or how the artifact is created. Just describe the visual abnormality in the region.


"""

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_masked}"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{original_target}"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{artifact_target}"},
                        {"type": "input_text", "text": f"{object_name}"},
                    ],
                },
            ],
            temperature=0.2,
            text_format=ArtifactExplanationResponse,
        )

        if money_manager:
            money_manager(response)

        return response.output_parsed

    except Exception as e:
        print(f"Error in artifact explanation from triplet ({artifact_type}): {e}")
        return {"explanation": ""}

def artifact_explanation(client, real_image, artifact_image, object_name, artifact_type, money_manager=None):
    """
    Generate natural language explanation of visual artifacts using OpenAI Vision API
    
    Args:
        client: OpenAI client
        real_image: Original image as numpy array with region visualized where artifact will be injected
        artifact_image: Modified image with artifact as numpy array with region visualized where artifact was injected
        object_name: Name of the object where artifact is applied
        artifact_type: Type of artifact ('addition', 'removal', 'distortion', 'fusion')
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        dict: Contains 'explanation' string and 'success' boolean
    """
    # Encode both images to base64
    base64_real_image = encode_image_to_base64(real_image)
    base64_artifact_image = encode_image_to_base64(artifact_image)
    
    # Create artifact-type-specific guidance (without explicitly mentioning the type)
    if artifact_type == 'addition':
        focus_guidance = "Pay attention to any duplicated parts, unnatural growths, or extra elements that conflict with normal anatomy or structure."
    elif artifact_type == 'removal':
        focus_guidance = "Pay attention to missing structure, unnatural gaps, smoothed-over areas, or anatomical discontinuity where something appears to be absent."
    elif artifact_type == 'distortion':
        focus_guidance = "Pay attention to warped shapes, unnatural geometry, irregular textures, or visual blending errors that make the structure appear broken or malformed."
    elif artifact_type == 'fusion':
        focus_guidance = "Pay attention to regions where two distinct parts or entities appear unnaturally merged together, with blurred boundaries, overlapped textures, or structural entanglement that makes separation implausible."
    else:
        focus_guidance = "Identify any visual abnormalities, unnatural features, or elements that appear incorrect or implausible."
    
    prompt = f"""
You are given two images:

- **Image A**: A real, original image, with a region visualized as red bounding box where the artifact is going to be injected.
- **Image B**: A modified version of the same scene, with a region visualized as green bounding box where the artifact is injected.

Here is the structured context:
- **Object Name**: {object_name}

Your task is to:
1. Examine the highlighted region in the **given image** (Image B).
2. Use your understanding of how the specified object should normally appear to identify abnormalities.
3. Write a natural language explanation describing what appears visually wrong or unnatural in the highlighted region.

**Do not mention or refer to the original image, the artifact type, or the image source explicitly.**
Base your explanation only on what is visible in the given image.

Focus your reasoning based on the artifact type (without stating it):

{focus_guidance}

Respond naturally and precisely, describing only what is visibly incorrect in the given image within the highlighted region.
"""

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_real_image}"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_artifact_image}"}
                    ]
                }
            ],
            temperature=0.2,
            text_format=ArtifactExplanationResponse
        )
        
        # Track costs with money manager
        if money_manager:
            money_manager(response)
        
        return {
            "success": True,
            "explanation": response.output_parsed.explanation
        }
            
    except Exception as e:
        print(f"Error in artifact explanation: {e}")
        return {
            "success": False,
            "explanation": "",
            "error": f"API error: {str(e)}"
        }
        
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
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens 

            if "o1" in self.model or "o3" in self.model or "o4" in self.model:
                output_tokens += (
                    response.usage.output_token_details.accepted_prediction_tokens
                    + response.usage.output_token_details.reasoning_tokens
                    + response.usage.output_token_details.rejected_prediction_tokens
                )

            input_cost = input_tokens / 1000 * self.input_cost
            output_cost = output_tokens / 1000 * self.output_cost

            self.total_cost += input_cost + output_cost

    def refresh(self) -> None:
        self.total_cost = 0.0########## deprecated ##########
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
