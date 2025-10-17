from dataclasses import dataclass
from typing import List, Tuple, Optional

BBox = Tuple[int, int, int, int]  # [xmin, ymin, xmax, ymax]

@dataclass
class ArtifactRegion:
    bbox: BBox
    label: str  # concise artifact description

@dataclass
class ArtiInstance:
    real_image: Optional[str]
    artifact_image: Optional[str]
    metadata_caption: Optional[str]
    artifacts: List[ArtifactRegion]

