"""
SatoLeo Count - End-to-End AI Pipeline
Coordinates detection, ByteTrack tracking, counting, and visual annotation.
"""
import time
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from backend.ai.config import PipelineConfig
from backend.ai.detector import FishDetector
from backend.ai.tracker import FishTracker
from backend.ai.counter import BaseCounter, create_counter

class FishCountingPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.detector = FishDetector(self.config)
        self.tracker = FishTracker(self.detector.model, self.config)
        self.counter: BaseCounter = create_counter(self.config)
        
        self.fps: float = 0.0
        self.frame_count: int = 0
        self._last_time = time.time()

    def reset(self):
        """Resets tracker, counter and pipeline statistics."""
        self.tracker.reset()
        self.counter.reset()
        self.frame_count = 0
        self.fps = 0.0
        self._last_time = time.time()

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single video frame:
        1. Detects fish in current frame.
        2. Tracks fish across frames using ByteTrack / spatial association.
        3. Updates the configured fish counter.
        4. Annotates the frame with bounding boxes, IDs, trajectories, and stats.
        Returns: (annotated_frame, metrics_dict)
        """
        if frame is None or frame.size == 0:
            return frame, {"error": "Invalid empty frame"}

        start_t = time.time()
        h, w = frame.shape[:2]
        
        # 1. Detection
        detections = self.detector.detect(frame)

        # 2. Tracking update
        tracked_objects = self.tracker.update(frame, detections)
        
        # 3. Counter update
        current_count = self.counter.update(tracked_objects, (h, w))
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - start_t
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

        # 4. Annotation
        annotated_frame = self._draw_annotations(frame.copy(), tracked_objects, current_count) if annotate else frame

        metrics = {
            "count": current_count,
            "active_tracks": len(tracked_objects),
            "fps": round(self.fps, 1),
            "frame_number": self.frame_count,
            "is_custom_model": self.detector.is_custom_model,
            "model_name": self.detector.model_name,
            "status": "counting",
            "counter_stats": self.counter.get_stats(),
            "tracks": [
                {
                    "id": obj["track_id"],
                    "bbox": [round(c, 1) for c in obj["bbox"]],
                    "conf": round(obj["confidence"], 2),
                    "centroid": obj["centroid"]
                }
                for obj in tracked_objects
            ]
        }

        return annotated_frame, metrics

    def _draw_annotations(self, frame: np.ndarray, tracks: List[Dict[str, Any]], count: int) -> np.ndarray:
        """
        Draws SatoLeo aquaculture styled annotations on the frame.
        - Cyan / Deep blue accents
        - Pill-shaped ID tags
        - Trajectory motion trails
        - Semi-transparent top HUD with big count and status
        """
        h, w = frame.shape[:2]

        # Draw ROI or Line Crossing guides if enabled
        if self.config.counting_mode == "line_crossing" and self.config.line_coords:
            p1 = (int(self.config.line_coords[0][0] * w), int(self.config.line_coords[0][1] * h))
            p2 = (int(self.config.line_coords[1][0] * w), int(self.config.line_coords[1][1] * h))
            cv2.line(frame, p1, p2, (0, 200, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "COUNT LINE", (p1[0], max(20, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Draw fish tracks
        for obj in tracks:
            t_id = obj["track_id"]
            bbox = obj["bbox"]
            conf = obj["confidence"]
            trajectory = obj.get("trajectory", [])
            x1, y1, x2, y2 = map(int, bbox)

            # Draw smooth trajectory trail (gradient fading)
            if len(trajectory) > 1:
                for idx in range(1, len(trajectory)):
                    thickness = int(np.sqrt(float(idx) / len(trajectory)) * 3) + 1
                    cv2.line(frame, trajectory[idx - 1], trajectory[idx], (255, 200, 0), thickness, cv2.LINE_AA)

            # Cyan bounding box (SatoLeo aquaculture brand)
            box_color = (245, 160, 0)  # BGR for SatoLeo cyan-blue #00a0f5
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Tag label: "Fish #ID (95%)"
            label = f"Fish #{t_id} ({int(conf * 100)}%)"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Label badge background
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 12, tag_y2), box_color, -1)
            cv2.putText(frame, label, (x1 + 6, tag_y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Centroid point
            cx, cy = obj["centroid"]
            cv2.circle(frame, (cx, cy), 4, (0, 255, 128), -1, cv2.LINE_AA)

        # Draw Top HUD Banner (Clean semi-transparent dark-blue bar)
        overlay = frame.copy()
        hud_height = 56
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # SatoLeo Brand & Count Text
        brand_text = "SATOLEO COUNT"
        cv2.putText(frame, brand_text, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 215, 255), 1, cv2.LINE_AA)
        
        model_tag = "CUSTOM SATOLEO AI" if self.detector.is_custom_model else "TEST / PREVIEW MODE"
        tag_color = (100, 230, 100) if self.detector.is_custom_model else (120, 180, 255)
        cv2.putText(frame, f"[{model_tag}]", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tag_color, 1, cv2.LINE_AA)

        # Total Count Highlight
        count_str = f"UNIQUE FISH: {count}"
        (cw, ch), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(frame, count_str, (w - cw - 20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS & In-frame counter in center
        info_str = f"In-frame: {len(tracks)} | FPS: {self.fps:.1f}"
        cv2.putText(frame, info_str, (int(w / 2) - 80, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1, cv2.LINE_AA)

        return frame
