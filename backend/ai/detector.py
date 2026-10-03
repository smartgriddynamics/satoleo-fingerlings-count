"""
SatoLeo Count - Fish Detection Module
Encapsulates Ultralytics YOLO model loading, inference, and structured detection output.
Includes adaptive computer vision detection for instant verification and fallback.
"""
import os
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from backend.ai.config import PipelineConfig

logger = logging.getLogger("satoleo.ai.detector")

class FishDetector:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.model_name: str = ""
        self.is_custom_model: bool = False
        self.has_yolo: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads custom trained SatoLeo model weights or fallback pretrained model.
        """
        resolved_path = self.config.resolve_model_path()
        self.is_custom_model = self.config.is_custom_model

        try:
            from ultralytics import YOLO
            self.has_yolo = True
            logger.info(f"Loading YOLO model from: {resolved_path} (Custom Model: {self.is_custom_model})")
            self.model = YOLO(resolved_path)
            self.model_name = os.path.basename(resolved_path)
            logger.info(f"YOLO Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.warning(f"Ultralytics YOLO unavailable or failed to load ({e}). Entering Computer Vision Fish Detector mode.")
            self.has_yolo = False
            self.model = None
            self.model_name = "cv_adaptive_detector (Preview Mode)"
            self.is_custom_model = False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single frame (BGR numpy array).
        Returns a list of detected objects:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str
            }, ...
        ]
        """
        if frame is None or frame.size == 0:
            return []

        if self.has_yolo and self.model is not None:
            try:
                device = None if self.config.device == "auto" else self.config.device
                results = self.model.predict(
                    source=frame,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    device=device,
                    verbose=False
                )
                
                detections = []
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        classes = boxes.cls.cpu().numpy().astype(int)
                        names = results[0].names or {}

                        for i in range(len(xyxy)):
                            cls_id = int(classes[i])
                            cls_name = "fish" if self.is_custom_model else names.get(cls_id, f"class_{cls_id}")
                            detections.append({
                                "bbox": [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])],
                                "confidence": float(confs[i]),
                                "class_id": cls_id,
                                "class_name": cls_name
                            })
                return detections
            except Exception as e:
                logger.error(f"YOLO Detection error: {e}")

        # Fallback CV fish segmentation detector (foreground & fish blob contrast)
        return self._cv_detect_fish(frame)

    def _cv_detect_fish(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Adaptive computer vision fish detector for fingerlings in containers/bowls.
        Identifies fish candidates by contrast, thresholding, and morphological filtering.
        """
        detections = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Otsu automatic thresholding for fish body contrast
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological opening to eliminate fine ripples
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = frame.shape[:2]
            min_area = 50
            max_area = (w * h) * 0.12

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bw) / bh if bh > 0 else 1.0
                    
                    # Ensure bounding box proportions fit fish geometry
                    if 0.15 < aspect_ratio < 6.5 and bw < w * 0.4 and bh < h * 0.4:
                        solidity = float(area) / (bw * bh + 1e-5)
                        conf = min(0.98, max(0.40, solidity))
                        if conf >= self.config.conf_threshold:
                            detections.append({
                                "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                                "confidence": round(conf, 2),
                                "class_id": 0,
                                "class_name": "fish"
                            })
        except Exception as e:
            logger.error(f"CV detect error: {e}")

        return detections

    def get_info(self) -> Dict[str, Any]:
        """Returns detector metadata."""
        return {
            "model_name": self.model_name,
            "is_custom_model": self.is_custom_model,
            "has_yolo": self.has_yolo,
            "conf_threshold": self.config.conf_threshold,
            "iou_threshold": self.config.iou_threshold,
            "device": self.config.device
        }
