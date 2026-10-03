"""
SatoLeo Count - Standalone Prediction Script
Tests YOLO detection on an image or video file.
Usage:
    python scripts/predict.py --source videos/sample_fish.mp4 --conf 0.25
"""
import argparse
import sys
import os
import cv2

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ai.detector import FishDetector
from backend.ai.config import PipelineConfig

def main():
    parser = argparse.ArgumentParser(description="SatoLeo Fish Detection CLI")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--weights", type=str, default="backend/models/satoleo_fish_best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display visual window")
    parser.add_argument("--output", type=str, default="videos/outputs/pred_result.jpg", help="Output save path")
    args = parser.parse_args()

    config = PipelineConfig(model_path=args.weights, conf_threshold=args.conf)
    detector = FishDetector(config)
    
    print("=" * 60)
    print(" SatoLeo Fish Detector Test ")
    print(f" Source: {args.source}")
    print(f" Model: {detector.model_name} (Custom: {detector.is_custom_model})")
    print(f" Confidence: {config.conf_threshold}")
    print("=" * 60)

    # Check if source is image
    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"Error: Could not read image at {args.source}")
            return
        
        detections = detector.detect(frame)
        print(f"Found {len(detections)} fish detections:")
        for i, det in enumerate(detections):
            print(f"  [{i+1}] {det['class_name']} - Conf: {det['confidence']:.2f}, Box: {det['bbox']}")
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 160, 0), 2)
            cv2.putText(frame, f"{det['class_name']} {det['confidence']:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, frame)
        print(f"Saved annotated result to {args.output}")

    else:
        # Video file
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video at {args.source}")
            return
        
        frame_idx = 0
        total_detections = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            dets = detector.detect(frame)
            total_detections += len(dets)
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: {len(dets)} fish detected in current frame")
        
        cap.release()
        print(f"Completed! Processed {frame_idx} frames. Total frame detections: {total_detections}")

if __name__ == "__main__":
    main()
