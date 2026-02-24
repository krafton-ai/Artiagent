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
            "explanation": "zebras have two ears, but as the target region contains an extra ear on the middle of its head, the zebra now has three ears, it is not naturally possible.",
            "label": "The zebra has an extra ear on the middle of its head.",
            "object_name": "an ear of zebra",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/positive/artifact_target.png'))
        },
        "negative": {
            "explanation": "Even though the artifact image has an manipulated version of the ear, the new ear has replaced the original one, which means that the artifact image does not have an artifact.",
            "label": "",
            "object_name": "an ear of a cat",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/negative/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/negative/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/addition/negative/artifact_target.png'))
        }
    },
    "removal": {
        "positive": {
            "explanation": "The bird is missing its tail, where the bird should have a tail.",
            "label": "The bird is missing its tail.",
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
            "explanation": "The facial features of the person are not defined well. The eyes and nose are not clearly visible.",
            "label": "The face of the person is distorted.",
            "object_name": "a hand of a person",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/distortion/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/distortion/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/distortion/positive/artifact_target.png'))
        }
    },
    "fusion": {
        "positive": {
            "explanation": "The boundary between the two cats is not visible and the head of the black cat seams to be merged into the white cat.",
            "label": "The heads of two cats are merged into one.",
            "object_name": "a cat and a cat",
            "original_masked": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/positive/original_masked.png')),
            "original_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/positive/original_target.png')),
            "artifact_target": encode_image_to_base64(Image.open('pipeline/in_context_exps/fusion/positive/artifact_target.png'))
        },
        "negative": {
            "explanation": "Although the boundary of the two elephant has been manipulated, the boundary between the two elephants is still visible in the artifact image.",
            "label": "",
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
    label: str
    
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
ㅊ
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
    "explanation": "Detailed description of what looks wrong in the region, without reasoning about the artifact type or how the artifact is created."
    "label": "Brief description of the artifact (empty string if no artifact)"
    
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
                            {"type": "output_text", "text": f'{{"has_artifact": true, "explanation": "{IN_CONTEXT_EXAMPLES["distortion"]["positive"]["explanation"]}", "label": "{IN_CONTEXT_EXAMPLES["distortion"]["positive"]["label"]}"}}'}
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
                            {"type": "output_text", "text": f'{{"has_artifact": true, "explanation": "{IN_CONTEXT_EXAMPLES[artifact_type]["positive"]["explanation"]}", "label": "{IN_CONTEXT_EXAMPLES[artifact_type]["positive"]["label"]}"}}'}
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
                            {"type": "output_text", "text": f'{{"has_artifact": false, "explanation": "{IN_CONTEXT_EXAMPLES[artifact_type]["negative"]["explanation"]}", "label": ""}}'}
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
            "Fusion artifacts occur when parts or two distinct entities are unnaturally merged together, "
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

def artifact_explanation(
    client,
    real_image,
    artifact_image,
    artifact_list: Optional[List[str]] = None,
    money_manager=None,
):
    """
    Generate a holistic explanation of why the artifact image is an artifact, using
    the provided list of injected artifact annotations.

    - Reads all entries in `artifact_list` with the form: "[xmin, xmax, ymin, ymax] <artifact description>".
    - Produces a concise, human-friendly explanation that leverages commonsense/anatomy
      (e.g., zebras normally have four legs) without mentioning coordinates.

    Returns a dict: {"success": bool, "explanation": str, "error": optional str}
    """

    # Encode images
    artifact_b64 = encode_image_to_base64(artifact_image)

    # Prepare artifact list text
    if artifact_list and isinstance(artifact_list, list) and len(artifact_list) > 0:
        cleaned_items = []
        for item in artifact_list:
            try:
                cleaned_items.append(re.sub(r"\s+", " ", str(item)).strip())
            except Exception:
                cleaned_items.append(str(item))
        artifact_items_text = "\n\n".join(f"- {c}" for c in cleaned_items)
    else:
        artifact_items_text = "(none provided)"

    system_prompt = f"""
You are an image artifact analyst. You will be given an image with artifacts and a list of injected artifact annotations, in the format of bbox:<(xmin, ymin, xmax, ymax)> description:<description of the artifact in that bbox region>.

Your job: Read ALL artifact annotations and write a single, holistic explanation the explanation of the anomalies in the image.
Guidance:
- Do NOT mention coordinates or the term "bbox". Use the annotations only to understand what is wrong.
- Use commonsense knowledge about typical anatomy/structure (e.g., zebras normally have four legs).
- If multiple issues appear, summarize the combined effect coherently rather than listing them mechanically.
- Keep it concise (2-3 sentences) and easy to understand. Avoid making concluding sentences. Just focus on explaining the abnormality.
"""

    user_content = [
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{artifact_b64}"},
        {
            "type": "input_text",
            "text": (
                "Injected artifact annotations (use for reasoning; do not mention coordinates):\n"
                f"{artifact_items_text}"
            ),
        },
    ]

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            text_format=ArtifactExplanationResponse,
        )

        if money_manager:
            money_manager(response)

        explanation = (
            response.output_parsed.explanation
            if response and hasattr(response, "output_parsed") and response.output_parsed
            else ""
        )
        explanation = explanation.strip()
        return explanation

    except Exception as e:
        print(f"Error in artifact_explanation: {e}")
        return None

        
def negative_explanation(
    client,
    real_image,
    artifact_image,
    artifact_explanation_text: str,
    money_manager=None,
):
    """
    Generate a concise explanation affirming that the real image is artifact-free,
    using the paired artifact image and its explanation as contrastive context.

    Args:
        client: OpenAI client-like object with responses.parse
        real_image: Clean image (PIL Image or ndarray)
        artifact_image: Artifacted counterpart (PIL Image or ndarray)
        artifact_explanation_text: Description of anomalies present in the artifact image
    """

    real_b64 = encode_image_to_base64(real_image)
    artifact_b64 = encode_image_to_base64(artifact_image)

    ctx_text = artifact_explanation_text if isinstance(artifact_explanation_text, str) else str(artifact_explanation_text)
    try:
        ctx_text = re.sub(r"\s+", " ", ctx_text).strip()
    except Exception:
        ctx_text = str(artifact_explanation_text)

    system_prompt = """
You are an image artifact analyst. You will be given:
1) A real (clean) image
2) Its paired artifacted image
3) A natural-language explanation describing the anomalies in the artifacted image

Your task is to write a concise explanation describing that the REAL image does not contain those anomalies.

Guidance:
- Focus on the real image; use the artifact explanation only to NEGATE anomalies (e.g., if it says a hand blends into a skateboard, confirm the hand does NOT blend and boundaries are clear).
- Affirm normal anatomy, counts, boundaries, and interactions (e.g., correct number of limbs/shoes, no extra parts, natural geometry).
- Do not mention coordinates, "bbox", model names, or refer to an "artifact image" explicitly; write a self-contained description of the real image.
- Tone: neutral and observational. Length: 2–3 sentences.
- Start with the phrase: "in the image". do not start with "in the real image".
"""

    user_content = [
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{real_b64}"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{artifact_b64}"},
        {
            "type": "input_text",
            "text": (
                "explanation (context only; NEGATE these anomalies in your description of the real image):\n"
                f"{ctx_text}"
            ),
        },
    ]

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            text_format=ArtifactExplanationResponse,
        )

        if money_manager:
            money_manager(response)

        explanation = (
            response.output_parsed.explanation
            if response and hasattr(response, "output_parsed") and response.output_parsed
            else ""
        )
        return explanation.strip()

    except Exception as e:
        print(f"Error in real_image_explanation_from_artifact_context: {e}")
        return None


def real_image_description(
    client,
    real_image,
    money_manager=None,
):
    """
    Generate a straightforward description of a real (clean) image without any artifact context.
    
    This function produces a neutral, observational description of what is visible in the image,
    focusing on the main subjects, their attributes, and their spatial relationships.
    
    Args:
        client: OpenAI client-like object with responses.parse
        real_image: Clean image (PIL Image or ndarray)
        money_manager: MoneyManager instance for cost tracking (optional)
        
    Returns:
        str: Natural language description of the real image, or None on error
    """
    
    real_b64 = encode_image_to_base64(real_image)
    
    system_prompt = """
You are an image description agent. You will be given a single image.

Your task is to write a concise, neutral description of what you observe in the image.

Guidance:
- Describe the main subjects/entities and their key attributes (e.g., appearance, pose, activity)
- Mention spatial relationships and context when relevant (e.g., foreground/background, relative positions)
- Focus on observable facts; avoid subjective interpretation or speculation
- Use clear, simple language that anyone can understand
- Tone: neutral and observational
- Length: 2–3 sentences
- Start with the phrase: "In the image". Do not use "In the real image".
- Do not mention image quality, artifacts, or abnormalities—simply describe what is present

Examples of good descriptions:
- "In the image, a brown dog sits on green grass with its tongue out. The dog appears to be a golden retriever, positioned in the center of the frame with trees visible in the background."
- "In the image, a person wearing a blue jacket stands near a red bicycle. The bicycle is parked against a white wall, and the person is holding the handlebars."
- "In the image, two cats are lying on a beige couch. One cat is black and white, while the other is orange, and both appear to be resting."
"""

    user_content = [
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{real_b64}"}
    ]

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            text_format=ArtifactExplanationResponse,
        )

        if money_manager:
            money_manager(response)

        description = (
            response.output_parsed.explanation
            if response and hasattr(response, "output_parsed") and response.output_parsed
            else ""
        )
        return description.strip()

    except Exception as e:
        print(f"Error in real_image_description: {e}")
        return None


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
        self.total_cost = 0.0