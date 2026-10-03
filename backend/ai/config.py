"""
SatoLeo Count - AI Pipeline Configuration
Handles model paths, thresholds, ByteTrack parameters, and counting modes.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class PipelineConfig:
    # Model configuration
    model_path: str = os.getenv("SATOLEO_MODEL_PATH", "backend/models/satoleo_fish_best.pt")
    fallback_model: str = "yolov8n.pt"
    conf_threshold: float = float(os.getenv("SATOLEO_CONF_THRESH", "0.25"))
    iou_threshold: float = float(os.getenv("SATOLEO_IOU_THRESH", "0.45"))
    target_classes: List[int] = field(default_factory=lambda: [0])  # Class 0 (custom fish model) or COCO classes
    device: str = os.getenv("SATOLEO_DEVICE", "auto")  # 'cpu', 'cuda', or 'auto'
    
    # Tracking configuration
    tracker_type: str = "bytetrack"
    tracker_config_path: str = "bytetrack.yaml"
    track_buffer: int = int(os.getenv("SATOLEO_TRACK_BUFFER", "30"))  # Frames to keep lost tracks
    min_hits: int = int(os.getenv("SATOLEO_MIN_HITS", "2"))  # Confirm unique fish after min_hits detections
    
    # Counting configuration
    counting_mode: str = os.getenv("SATOLEO_COUNT_MODE", "unique_id")  # 'unique_id', 'line_crossing', 'roi'
    
    # Line crossing parameters: [[x1, y1], [x2, y2]] in normalized [0, 1] coords
    line_coords: Optional[List[Tuple[float, float]]] = field(
        default_factory=lambda: [(0.1, 0.5), (0.9, 0.5)]
    )
    
    # ROI polygon: list of normalized (x, y) coordinates
    roi_polygon: Optional[List[Tuple[float, float]]] = None
    
    # Maximum trajectory history length to keep for smooth visualization
    max_trajectory_length: int = 30
    
    # Running state info
    is_custom_model: bool = False
    
    def resolve_model_path(self) -> str:
        """
        Check if custom trained SatoLeo weights exist; if not, return fallback model.
        """
        if os.path.exists(self.model_path):
            self.is_custom_model = True
            return self.model_path
        self.is_custom_model = False
        return self.fallback_model
