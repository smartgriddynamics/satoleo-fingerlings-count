"""
SatoLeo Count - Counting Module
Modular, extensible fish counting strategies:
1. UniqueIdCounter: Counts distinct validated track IDs over time.
2. LineCrossingCounter: Counts fish crossing a virtual gate/line.
3. RoiCounter: Counts fish entering or residing within a Polygon Region of Interest.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from backend.ai.config import PipelineConfig

logger = logging.getLogger("satoleo.ai.counter")

class BaseCounter(ABC):
    """Abstract base class for all fish counting strategies."""
    
    @abstractmethod
    def update(self, tracked_objects: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> int:
        """
        Updates the counter state with new tracked objects.
        Returns the current cumulative fish count.
        """
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Returns the current cumulative count."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the counter state."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Returns structured counting statistics."""
        pass


class UniqueIdCounter(BaseCounter):
    """
    Counts each unique fish ID that has been reliably tracked for at least `min_hits` frames.
    Prevents false counting from brief 1-frame detection noise.
    """
    def __init__(self, min_hits: int = 2):
        self.min_hits = min_hits
        self.confirmed_ids: Set[int] = set()
        self.candidate_ids: Dict[int, int] = {}  # track_id -> hit count
        self.counted_timeline: List[Dict[str, Any]] = []

    def update(self, tracked_objects: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> int:
        for obj in tracked_objects:
            t_id = obj["track_id"]
            hits = obj.get("hits", 1)
            
            if t_id not in self.confirmed_ids:
                if hits >= self.min_hits:
                    self.confirmed_ids.add(t_id)
                    self.counted_timeline.append({
                        "track_id": t_id,
                        "confidence": obj.get("confidence", 0.0),
                        "centroid": obj.get("centroid", (0, 0))
                    })
                    logger.debug(f"New unique fish confirmed! ID: {t_id}, Total: {len(self.confirmed_ids)}")
        return len(self.confirmed_ids)

    def get_count(self) -> int:
        return len(self.confirmed_ids)

    def reset(self) -> None:
        self.confirmed_ids.clear()
        self.candidate_ids.clear()
        self.counted_timeline.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": "unique_id",
            "total_count": len(self.confirmed_ids),
            "confirmed_ids": sorted(list(self.confirmed_ids)),
            "min_hits_threshold": self.min_hits
        }


class LineCrossingCounter(BaseCounter):
    """
    Counts fish when their trajectory crosses a specified virtual line.
    Useful for counting fingerlings flowing through a channel or pipe.
    """
    def __init__(self, line_coords: Optional[List[Tuple[float, float]]] = None):
        # Normalized coordinates: [(x1, y1), (x2, y2)]
        self.line_coords = line_coords or [(0.1, 0.5), (0.9, 0.5)]
        self.crossed_ids: Set[int] = set()
        self.count_in: int = 0
        self.count_out: int = 0

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A, B, C, D):
        """Returns True if line segment AB intersects line segment CD."""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)

    def update(self, tracked_objects: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> int:
        h, w = frame_shape
        # Convert normalized line to pixel coordinates
        p1 = (int(self.line_coords[0][0] * w), int(self.line_coords[0][1] * h))
        p2 = (int(self.line_coords[1][0] * w), int(self.line_coords[1][1] * h))

        for obj in tracked_objects:
            t_id = obj["track_id"]
            history = obj.get("trajectory", [])
            
            if len(history) >= 2 and t_id not in self.crossed_ids:
                prev_pt = history[-2]
                curr_pt = history[-1]
                
                if self._intersect(prev_pt, curr_pt, p1, p2):
                    self.crossed_ids.add(t_id)
                    # Determine crossing direction
                    dy = curr_pt[1] - prev_pt[1]
                    if dy > 0:
                        self.count_in += 1
                    else:
                        self.count_out += 1
        return len(self.crossed_ids)

    def get_count(self) -> int:
        return len(self.crossed_ids)

    def reset(self) -> None:
        self.crossed_ids.clear()
        self.count_in = 0
        self.count_out = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": "line_crossing",
            "total_count": len(self.crossed_ids),
            "count_in": self.count_in,
            "count_out": self.count_out,
            "crossed_ids": sorted(list(self.crossed_ids))
        }


class RoiCounter(BaseCounter):
    """
    Counts fish entering or currently located within a defined Region of Interest (ROI).
    """
    def __init__(self, polygon: Optional[List[Tuple[float, float]]] = None):
        # Normalized coordinates polygon
        self.polygon = polygon or [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        self.counted_ids: Set[int] = set()
        self.current_in_roi: int = 0

    def _point_in_polygon(self, pt: Tuple[int, int], poly: List[Tuple[int, int]]) -> bool:
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def update(self, tracked_objects: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> int:
        h, w = frame_shape
        pixel_poly = [(int(pt[0] * w), int(pt[1] * h)) for pt in self.polygon]
        
        in_roi_now = 0
        for obj in tracked_objects:
            cx, cy = obj["centroid"]
            if self._point_in_polygon((cx, cy), pixel_poly):
                in_roi_now += 1
                self.counted_ids.add(obj["track_id"])

        self.current_in_roi = in_roi_now
        return len(self.counted_ids)

    def get_count(self) -> int:
        return len(self.counted_ids)

    def reset(self) -> None:
        self.counted_ids.clear()
        self.current_in_roi = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": "roi",
            "total_unique_in_roi": len(self.counted_ids),
            "current_in_roi": self.current_in_roi,
            "counted_ids": sorted(list(self.counted_ids))
        }


def create_counter(config: PipelineConfig) -> BaseCounter:
    """Factory function to instantiate the configured counting strategy."""
    mode = config.counting_mode.lower()
    if mode == "line_crossing":
        return LineCrossingCounter(line_coords=config.line_coords)
    elif mode == "roi":
        return RoiCounter(polygon=config.roi_polygon)
    else:
        return UniqueIdCounter(min_hits=config.min_hits)
