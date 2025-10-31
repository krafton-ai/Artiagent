import random
from typing import Tuple

class VQAPrompts:
    """Question templates for VQA generation."""
    # Toggle: when False (default), always use the first prompt in each variant list
    # When True, randomly sample from variants
    USE_VARIANTS = False
    
    # 1.1 Binary Detection (evaluated - strict JSON)
    BINARY_DETECTION_VARIANTS = [
        "Does this image contain any visual artifacts?",
        "Are there any artifacts present in this image?",
        "Can you detect any visual artifacts in this image?"
    ]
    
    BINARY_DETECTION_FORMAT = """
Answer format (strict JSON):
{"type":"binary_detection","artifact_present":"yes|no"}"""
    
    # 1.2 Localization (evaluated - strict JSON)
    LOCALIZATION_VARIANTS = [
        "Provide pixel bounding boxes for ALL artifact regions in this image.",
        "Locate all artifact regions using pixel coordinates.",
        "Identify the bounding boxes of all artifacts in this image."
    ]
    
    LOCALIZATION_FORMAT = """
Answer format (strict JSON):
{"type":"localization","coord_space":"pixel","bboxes":[{"bbox":[xmin,ymin,xmax,ymax]}, ...]}"""
    
    # 1.3 Global Explanation (evaluated - strict JSON)
    GLOBAL_EXPLANATION_VARIANTS = [
        "Describe how the artifacts appear in this image.",
        "Explain the visual characteristics of the artifacts in this image.",
        "What artifacts can you observe in this image and how do they appear?"
    ]
    
    GLOBAL_EXPLANATION_FORMAT = """
Answer format (strict JSON):
{"type":"global_explanation","explanations":["<short description>"]}"""
    
    # 1.5 Global Explanation Real (training-only - natural text)
    GLOBAL_EXPLANATION_REAL_VARIANTS = [
        "Explain why this image is free of artifacts.",
        "Why does this image not contain any artifacts?",
        "What makes this image artifact-free?"
    ]
    
    # 1.4 Regional Explanation (training-only - natural text)
    REGIONAL_EXPLANATION_VARIANTS = [
        "For region [{xmin},{ymin},{xmax},{ymax}], explain why it is considered an artifact.",
        "Explain why region [{xmin},{ymin},{xmax},{ymax}] contains an artifact.",
        "What makes the region [{xmin},{ymin},{xmax},{ymax}] an artifact?"
    ]
    
    # 4.1 Binary (Pair) - with artifact-injected context
    PAIR_BINARY_TEMPLATES = [
        "One is a real image and the other is an artifact-injected version.\nWhich image contains artifacts? Answer 'first' or 'second'."
    ]
    
    # 4.2 Localization (Pair) - context varies by image order
    PAIR_LOCALIZATION_FIRST_ARTIFACT = [
        "The first image is an artifact-injected version of the second image.\nLocate the artifact region(s) in the first image."
    ]
    PAIR_LOCALIZATION_SECOND_ARTIFACT = [
        "The second image is an artifact-injected version of the first image.\nLocate the artifact region(s) in the second image."
    ]
    
    # 4.3 Regional (Pair)
    PAIR_REGIONAL_FIRST_ARTIFACT = [
        "The first image is an artifact-injected version of the second image.\nDescribe what kinds of artifacts are visible and where they appear in the first image."
    ]
    PAIR_REGIONAL_SECOND_ARTIFACT = [
        "The second image is an artifact-injected version of the first image.\nDescribe what kinds of artifacts are visible and where they appear in the second image."
    ]
    
    # 4.4 Explanation (Pair)
    PAIR_EXPLANATION_FIRST_ARTIFACT = [
        "The first image is an artifact-injected version of the second image.\nExplain how the artifact-injected image differs from the real one."
    ]
    PAIR_EXPLANATION_SECOND_ARTIFACT = [
        "The second image is an artifact-injected version of the first image.\nExplain how the artifact-injected image differs from the real one."
    ]
    
    @staticmethod
    def get_binary_detection(include_format: bool = True) -> str:
        """Get binary detection question."""
        q = (
            random.choice(VQAPrompts.BINARY_DETECTION_VARIANTS)
            if VQAPrompts.USE_VARIANTS
            else VQAPrompts.BINARY_DETECTION_VARIANTS[0]
        )
        if include_format:
            q += VQAPrompts.BINARY_DETECTION_FORMAT
        return q
    
    @staticmethod
    def get_localization(include_format: bool = True) -> str:
        """Get localization question."""
        q = (
            random.choice(VQAPrompts.LOCALIZATION_VARIANTS)
            if VQAPrompts.USE_VARIANTS
            else VQAPrompts.LOCALIZATION_VARIANTS[0]
        )
        if include_format:
            q += VQAPrompts.LOCALIZATION_FORMAT
        return q
    
    @staticmethod
    def get_global_explanation(include_format: bool = True) -> str:
        """Get global explanation question."""
        q = (
            random.choice(VQAPrompts.GLOBAL_EXPLANATION_VARIANTS)
            if VQAPrompts.USE_VARIANTS
            else VQAPrompts.GLOBAL_EXPLANATION_VARIANTS[0]
        )
        if include_format:
            q += VQAPrompts.GLOBAL_EXPLANATION_FORMAT
        return q
    
    @staticmethod
    def get_regional_explanation(bbox: Tuple[int, int, int, int]) -> str:
        """Get regional explanation question for specific bbox."""
        template = (
            random.choice(VQAPrompts.REGIONAL_EXPLANATION_VARIANTS)
            if VQAPrompts.USE_VARIANTS
            else VQAPrompts.REGIONAL_EXPLANATION_VARIANTS[0]
        )
        xmin, ymin, xmax, ymax = bbox
        return template.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    
    @staticmethod
    def get_pair_binary() -> str:
        """Get pair binary question."""
        return random.choice(VQAPrompts.PAIR_BINARY_TEMPLATES)
    
    @staticmethod
    def get_pair_localization(artifact_position: str) -> str:
        """Get pair localization question.
        
        Args:
            artifact_position: Either 'first' or 'second'
        """
        if artifact_position == "first":
            return (
                random.choice(VQAPrompts.PAIR_LOCALIZATION_FIRST_ARTIFACT)
                if VQAPrompts.USE_VARIANTS
                else VQAPrompts.PAIR_LOCALIZATION_FIRST_ARTIFACT[0]
            )
        else:
            return (
                random.choice(VQAPrompts.PAIR_LOCALIZATION_SECOND_ARTIFACT)
                if VQAPrompts.USE_VARIANTS
                else VQAPrompts.PAIR_LOCALIZATION_SECOND_ARTIFACT[0]
            )
    
    @staticmethod
    def get_pair_regional(artifact_position: str) -> str:
        """Get pair regional question.
        
        Args:
            artifact_position: Either 'first' or 'second'
        """
        if artifact_position == "first":
            return (
                random.choice(VQAPrompts.PAIR_REGIONAL_FIRST_ARTIFACT)
                if VQAPrompts.USE_VARIANTS
                else VQAPrompts.PAIR_REGIONAL_FIRST_ARTIFACT[0]
            )
        else:
            return (
                random.choice(VQAPrompts.PAIR_REGIONAL_SECOND_ARTIFACT)
                if VQAPrompts.USE_VARIANTS
                else VQAPrompts.PAIR_REGIONAL_SECOND_ARTIFACT[0]
            )
    
    @staticmethod
    def get_pair_explanation(artifact_position: str) -> str:
        """Get pair explanation question.
        
        Args:
            artifact_position: Either 'first' or 'second'
        """
        if artifact_position == "first":
            return (
                random.choice(VQAPrompts.PAIR_EXPLANATION_FIRST_ARTIFACT)
                if VQAPrompts.USE_VARIANTS
                else VQAPrompts.PAIR_EXPLANATION_FIRST_ARTIFACT[0]
            )
        else:
            return (
                random.choice(VQAPrompts.PAIR_EXPLANATION_SECOND_ARTIFACT)
                if VQAPrompts.USE_VARIANTS
                else VQAPrompts.PAIR_EXPLANATION_SECOND_ARTIFACT[0]
            )
    
    @staticmethod
    def get_global_explanation_real() -> str:
        """Get global explanation question for real images (artifact-free)."""
        return (
            random.choice(VQAPrompts.GLOBAL_EXPLANATION_REAL_VARIANTS)
            if VQAPrompts.USE_VARIANTS
            else VQAPrompts.GLOBAL_EXPLANATION_REAL_VARIANTS[0]
        )

