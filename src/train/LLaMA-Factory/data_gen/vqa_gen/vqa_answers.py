import json
import random
from typing import List, Tuple
from .types import ArtifactRegion

class VQAAnswers:
    """Answer generation for VQA tasks."""
    
    @staticmethod
    def binary_detection(has_artifacts: bool) -> str:
        """Generate binary detection answer (strict JSON)."""
        return json.dumps({
            "type": "binary_detection",
            "artifact_present": "yes" if has_artifacts else "no"
        }, ensure_ascii=False)
    
    @staticmethod
    def localization(artifacts: List[ArtifactRegion]) -> str:
        """Generate localization answer (strict JSON)."""
        bboxes = [{"bbox": list(artifact.bbox)} for artifact in artifacts]
        return json.dumps({
            "type": "localization",
            "coord_space": "pixel",
            "bboxes": bboxes
        }, ensure_ascii=False)
    
    @staticmethod
    def global_explanation(metadata_caption: str = None, artifacts: List[ArtifactRegion] = None) -> str:
        """Generate global explanation answer (strict JSON).
        
        Args:
            metadata_caption: Caption from metadata (preferred)
            artifacts: List of artifact regions (fallback)
        """
        if metadata_caption:
            explanations = [metadata_caption]
        elif artifacts:
            explanations = [artifact.label for artifact in artifacts]
        else:
            raise ValueError("Global explanation requires a caption or artifacts.")
        
        return json.dumps({
            "type": "global_explanation",
            "explanations": explanations
        }, ensure_ascii=False)
    
    @staticmethod
    def regional_explanation(artifact: ArtifactRegion) -> str:
        """Generate regional explanation answer (natural text)."""
        xmin, ymin, xmax, ymax = artifact.bbox

        return f"in the bbox region [{xmin},{ymin},{xmax},{ymax}], {artifact.label}"
    
    @staticmethod
    def pair_binary(artifact_position: str) -> str:
        """Generate pair binary answer (single token).
        
        Args:
            artifact_position: Either 'first' or 'second'
        
        Returns:
            'first' or 'second'
        """
        return artifact_position
    
    @staticmethod
    def pair_localization(artifacts: List[ArtifactRegion]) -> str:
        """Generate pair localization answer.
        
        Args:
            artifacts: List of artifact regions
        
        Returns:
            'there are artifact(s) in [x,y,x,y] and [x,y,x,y]'
        """
        # Sort by (ymin, xmin) for consistent ordering
        sorted_artifacts = sorted(artifacts, key=lambda a: (a.bbox[1], a.bbox[0]))
        bbox_strs = [f"[{a.bbox[0]},{a.bbox[1]},{a.bbox[2]},{a.bbox[3]}]" for a in sorted_artifacts]
        return f"there are artifact(s) in {' and '.join(bbox_strs)}"
    
    @staticmethod
    def pair_regional(artifacts: List[ArtifactRegion]) -> str:
        """Generate pair regional answer.
        
        Args:
            artifacts: List of artifact regions
        
        Returns:
            'there is <label> in [bbox] and there is <label> in [bbox]'
        """
        # Sort by (ymin, xmin) for consistent ordering
        sorted_artifacts = sorted(artifacts, key=lambda a: (a.bbox[1], a.bbox[0]))
        parts = [f"there is {a.label} in [{a.bbox[0]},{a.bbox[1]},{a.bbox[2]},{a.bbox[3]}]" 
                 for a in sorted_artifacts]
        return " and ".join(parts)
    
    @staticmethod
    def pair_explanation(metadata_caption: str) -> str:
        """Generate pair explanation answer.
        
        Args:
            metadata_caption: Caption from metadata
        
        Returns:
            'there is <caption>'
        """
        return f"there is {metadata_caption}"

