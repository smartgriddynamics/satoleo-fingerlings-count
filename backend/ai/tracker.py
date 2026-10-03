"""
SatoLeo Count - Fish Tracker Module
Integrates ByteTrack tracking with persistent ID management and motion trajectories.
Provides spatial IoU + Centroid association fallback for test environments.
"""
import logging
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from backend.ai.config import PipelineConfig

logger = logging.getLogger("satoleo.ai.tracker")

class TrackedObject:
    def __init__(self, track_id: int, bbox: List[float], confidence: float, class_name: str = "fish"):
        self.track_id: int = track_id
        self.bbox: List[float] = bbox
        self.confidence: float = confidence
        self.class_name: str = class_name
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0
        self.history: List[Tuple[int, int]] = []
        self._update_centroid()

    def _update_centroid(self):
        cx = int((self.bbox[0] + self.bbox[2]) / 2.0)
        cy = int((self.bbox[1] + self.bbox[3]) / 2.0)
        self.centroid = (cx, cy)
        self.history.append(self.centroid)

    def update(self, bbox: List[float], confidence: float, max_len: int = 30):
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self._update_centroid()
        if len(self.history) > max_len:
            self.history.pop(0)

class FishTracker:
    def __init__(self, detector_model: Any = None, config: Optional[PipelineConfig] = None):
        self.model = detector_model
        self.config = config or PipelineConfig()
        self.tracks_dict: Dict[int, TrackedObject] = {}
        self.active_tracks: List[Dict[str, Any]] = []
        self.next_id: int = 1
        self.max_distance: float = 65.0  # Max pixel distance between frames for same fish

    def reset(self):
        """Resets all active tracking memory."""
        self.tracks_dict.clear()
        self.active_tracks.clear()
        self.next_id = 1
        logger.info("Tracker state reset.")

    def update(self, frame: np.ndarray, detections: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Maintains consistent IDs across sequential frames.
        Returns a list of active tracked items in current frame.
        """
        if frame is None or frame.size == 0:
            return []

        # If Ultralytics model with ByteTrack is available:
        if self.model is not None and hasattr(self.model, 'track'):
            try:
                device = None if self.config.device == "auto" else self.config.device
                results = self.model.track(
                    source=frame,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    tracker=self.config.tracker_config_path,
                    persist=True,
                    device=device,
                    verbose=False
                )

                current_frame_tracks = []
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and boxes.is_track:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        
                        for i in range(len(track_ids)):
                            t_id = int(track_ids[i])
                            bbox = [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]
                            conf = float(confs[i])
                            
                            if t_id not in self.tracks_dict:
                                self.tracks_dict[t_id] = TrackedObject(
                                    track_id=t_id,
                                    bbox=bbox,
                                    confidence=conf
                                )
                            else:
                                self.tracks_dict[t_id].update(
                                    bbox=bbox,
                                    confidence=conf,
                                    max_len=self.config.max_trajectory_length
                                )

                            obj = self.tracks_dict[t_id]
                            current_frame_tracks.append({
                                "track_id": t_id,
                                "bbox": bbox,
                                "confidence": conf,
                                "centroid": obj.centroid,
                                "hits": obj.hits,
                                "trajectory": list(obj.history)
                            })
                
                self.active_tracks = current_frame_tracks
                return current_frame_tracks
            except Exception as e:
                logger.debug(f"YOLO ByteTrack fallback to spatial tracker: {e}")

        # Fallback spatial centroid + IoU tracker
        return self._spatial_track(detections or [])

    def _spatial_track(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Associates detections to existing tracks using distance and spatial proximity.
        """
        # Increment time_since_update for all existing tracks
        for obj in self.tracks_dict.values():
            obj.time_since_update += 1

        matched_tracks = set()
        matched_dets = set()
        current_frame_tracks = []

        # Find best matches based on centroid distance
        if len(self.tracks_dict) > 0 and len(detections) > 0:
            track_ids = list(self.tracks_dict.keys())
            for d_idx, det in enumerate(detections):
                bbox = det["bbox"]
                dcx = (bbox[0] + bbox[2]) / 2.0
                dcy = (bbox[1] + bbox[3]) / 2.0

                best_id = None
                best_dist = float('inf')

                for t_id in track_ids:
                    if t_id in matched_tracks:
                        continue
                    obj = self.tracks_dict[t_id]
                    if obj.time_since_update > self.config.track_buffer:
                        continue
                    tcx, tcy = obj.centroid
                    dist = math.hypot(dcx - tcx, dcy - tcy)
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_id = t_id

                if best_id is not None:
                    matched_tracks.add(best_id)
                    matched_dets.add(d_idx)
                    self.tracks_dict[best_id].update(
                        bbox=bbox,
                        confidence=det["confidence"],
                        max_len=self.config.max_trajectory_length
                    )
                    obj = self.tracks_dict[best_id]
                    current_frame_tracks.append({
                        "track_id": best_id,
                        "bbox": bbox,
                        "confidence": det["confidence"],
                        "centroid": obj.centroid,
                        "hits": obj.hits,
                        "trajectory": list(obj.history)
                    })

        # Register new detections as new unique fish tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_id = self.next_id
                self.next_id += 1
                new_obj = TrackedObject(
                    track_id=new_id,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    class_name=det.get("class_name", "fish")
                )
                self.tracks_dict[new_id] = new_obj
                current_frame_tracks.append({
                    "track_id": new_id,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "centroid": new_obj.centroid,
                    "hits": new_obj.hits,
                    "trajectory": list(new_obj.history)
                })

        # Purge stale tracks beyond track_buffer
        to_delete = [
            t_id for t_id, obj in self.tracks_dict.items()
            if obj.time_since_update > self.config.track_buffer
        ]
        for t_id in to_delete:
            del self.tracks_dict[t_id]

        self.active_tracks = current_frame_tracks
        return current_frame_tracks
