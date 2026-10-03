"""
SatoLeo Count - Standalone Video Tracking & Unique Fish Counting CLI
Executes full OpenCV + YOLO + ByteTrack + Unique ID Counting Pipeline on a video.

Usage:
    python scripts/track.py --source videos/sample_fish.mp4 --output videos/outputs/tracked_fish.mp4
"""
import argparse
import sys
import os
import cv2
import time

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ai.pipeline import FishCountingPipeline
from backend.ai.config import PipelineConfig

def main():
    parser = argparse.ArgumentParser(description="SatoLeo Fish Tracking & Counting CLI")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="videos/outputs/tracked_fish.mp4", help="Path for output video")
    parser.add_argument("--weights", type=str, default="backend/models/satoleo_fish_best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--mode", type=str, default="unique_id", choices=["unique_id", "line_crossing", "roi"], help="Counting mode")
    parser.add_argument("--min-hits", type=int, default=2, help="Min detections before confirming unique fish ID")
    parser.add_argument("--show", action="store_true", help="Display realtime OpenCV window")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Input video not found at {args.source}")
        sys.exit(1)

    config = PipelineConfig(
        model_path=args.weights,
        conf_threshold=args.conf,
        counting_mode=args.mode,
        min_hits=args.min_hits
    )

    print("=" * 65)
    print(" SATOLEO COUNT — AI TRACKING & COUNTING ENGINE ")
    print(f" Input Video   : {args.source}")
    print(f" Output Video  : {args.output}")
    print(f" Model Selected: {config.resolve_model_path()}")
    print(f" Custom Model  : {config.is_custom_model}")
    print(f" Counting Mode : {args.mode}")
    print(f" Conf Threshold: {args.conf}")
    print(f" Min Hits/ID   : {args.min_hits}")
    print("=" * 65)

    pipeline = FishCountingPipeline(config)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")

    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated_frame, metrics = pipeline.process_frame(frame, annotate=True)
            out.write(annotated_frame)

            if args.show:
                cv2.imshow("SatoLeo Count - AI Tracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user.")
                    break

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"Frame [{frame_idx}/{total_frames}] ({progress:.1f}%) | Count: {metrics['count']} fish | In-frame: {metrics['active_tracks']} | FPS: {metrics['fps']:.1f}")

    finally:
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print("\n" + "=" * 65)
    print(" PROCESSING COMPLETED ")
    print(f" Total Frames Processed: {frame_idx}")
    print(f" Elapsed Time          : {total_time:.2f} seconds")
    print(f" Average Processing FPS: {avg_fps:.1f}")
    print(f" FINAL UNIQUE FISH COUNT: {pipeline.counter.get_count()}")
    print(f" Output Video Saved to : {args.output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
