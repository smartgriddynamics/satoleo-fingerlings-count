"""
SatoLeo Count - API Data Models (Pydantic schemas)
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "SatoLeo Count AI Engine"

class TrackItem(BaseModel):
    id: int
    bbox: List[float]
    conf: float
    centroid: Tuple[int, int]

class CountStatusResponse(BaseModel):
    count: int = 0
    active_tracks: int = 0
    fps: float = 0.0
    status: str = "idle"  # idle, counting, paused, stopped
    model_name: str = "yolov8n.pt"
    is_custom_model: bool = False
    counting_mode: str = "unique_id"
    counter_stats: Dict[str, Any] = Field(default_factory=dict)
    recent_tracks: List[TrackItem] = Field(default_factory=list)

class StartCountRequest(BaseModel):
    mode: Optional[str] = "unique_id"
    conf_threshold: Optional[float] = 0.25
    min_hits: Optional[int] = 2
    model_path: Optional[str] = None

class VideoProcessResponse(BaseModel):
    status: str = "completed"
    filename: str
    total_frames: int
    final_count: int
    avg_fps: float
    elapsed_time: float
    output_video_url: Optional[str] = None
    is_custom_model: bool
    model_name: str
    counter_stats: Dict[str, Any]

class UpdateSettingsRequest(BaseModel):
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    counting_mode: Optional[str] = None
    min_hits: Optional[int] = None
    track_buffer: Optional[int] = None
