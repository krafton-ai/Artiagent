from typing import List, Tuple, Dict, Any, Optional
from .types import ArtiInstance, ArtifactRegion
from .vqa_prompts import VQAPrompts
from .vqa_answers import VQAAnswers

class QAPair:
    """A single question-answer pair."""
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

class VQABuilders:
    """Build Q-A pairs for different task types."""
    
    @staticmethod
    def build_binary_detection_real(format_dropout: float = 0.15) -> QAPair:
        """Build binary detection Q-A for real image (no artifacts)."""
        include_format = random.random() > format_dropout
        question = VQAPrompts.get_binary_detection(include_format=include_format)
        answer = VQAAnswers.binary_detection(has_artifacts=False)
        return QAPair(question, answer)
    
    @staticmethod
    def build_binary_detection_artifact(format_dropout: float = 0.15) -> QAPair:
        """Build binary detection Q-A for artifact image (has artifacts)."""
        include_format = random.random() > format_dropout
        question = VQAPrompts.get_binary_detection(include_format=include_format)
        answer = VQAAnswers.binary_detection(has_artifacts=True)
        return QAPair(question, answer)
    
    @staticmethod
    def build_localization(artifacts: List[ArtifactRegion], format_dropout: float = 0.15) -> QAPair:
        """Build localization Q-A for artifact image."""
        include_format = random.random() > format_dropout
        question = VQAPrompts.get_localization(include_format=include_format)
        answer = VQAAnswers.localization(artifacts)
        return QAPair(question, answer)
    
    @staticmethod
    def build_global_explanation(metadata_caption: str = None, artifacts: List[ArtifactRegion] = None, format_dropout: float = 0.15) -> QAPair:
        """Build global explanation Q-A for artifact image.
        
        Args:
            metadata_caption: Caption from metadata (preferred)
            artifacts: List of artifact regions (fallback)
            format_dropout: Probability to omit format instructions
        """
        include_format = random.random() > format_dropout
        question = VQAPrompts.get_global_explanation(include_format=include_format)
        answer = VQAAnswers.global_explanation(metadata_caption=metadata_caption, artifacts=artifacts)
        return QAPair(question, answer)
    
    @staticmethod
    def build_regional_explanation(artifact: ArtifactRegion) -> QAPair:
        """Build regional explanation Q-A for specific artifact region."""
        question = VQAPrompts.get_regional_explanation(artifact.bbox)
        answer = VQAAnswers.regional_explanation(artifact)
        return QAPair(question, answer)
    
    @staticmethod
    def build_pair_binary(artifact_position: str) -> QAPair:
        """Build pair binary Q-A (4.1).
        
        Args:
            artifact_position: Either 'first' or 'second'
        """
        question = VQAPrompts.get_pair_binary()
        answer = VQAAnswers.pair_binary(artifact_position)
        return QAPair(question, answer)
    
    @staticmethod
    def build_pair_localization(artifacts: List[ArtifactRegion], artifact_position: str) -> Optional[QAPair]:
        """Build pair localization Q-A (4.2).
        
        Args:
            artifacts: List of artifact regions
            artifact_position: Either 'first' or 'second'
        
        Returns:
            QAPair or None if no artifacts
        """
        if not artifacts:
            return None
        question = VQAPrompts.get_pair_localization(artifact_position)
        answer = VQAAnswers.pair_localization(artifacts)
        return QAPair(question, answer)
    
    @staticmethod
    def build_pair_regional(artifacts: List[ArtifactRegion], artifact_position: str) -> Optional[QAPair]:
        """Build pair regional Q-A (4.3).
        
        Args:
            artifacts: List of artifact regions
            artifact_position: Either 'first' or 'second'
        
        Returns:
            QAPair or None if no artifacts
        """
        if not artifacts:
            return None
        question = VQAPrompts.get_pair_regional(artifact_position)
        answer = VQAAnswers.pair_regional(artifacts)
        return QAPair(question, answer)
    
    @staticmethod
    def build_pair_explanation(metadata_caption: str, artifact_position: str) -> Optional[QAPair]:
        """Build pair explanation Q-A (4.4).
        
        Args:
            metadata_caption: Caption from metadata
            artifact_position: Either 'first' or 'second'
        
        Returns:
            QAPair or None if no caption
        """
        if not metadata_caption:
            return None
        question = VQAPrompts.get_pair_explanation(artifact_position)
        answer = VQAAnswers.pair_explanation(metadata_caption)
        return QAPair(question, answer)
    
    @staticmethod
    def build_global_explanation_real(real_caption: str) -> QAPair:
        """Build global explanation Q-A for real image (artifact-free).
        
        Args:
            real_caption: Description of why the image is artifact-free
        
        Returns:
            QAPair with natural text answer
        """
        question = VQAPrompts.get_global_explanation_real()
        answer = real_caption
        return QAPair(question, answer)

import random

