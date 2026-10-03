"""
SatoLeo Count - Video Frame Extractor
Extracts sharp, evenly spaced frames from fish bowl/tank recordings for annotation in CVAT.

Usage:
    python scripts/extract_frames.py --video videos/recording1.mp4 --output dataset/raw_frames --interval 15
"""
import argparse
import os
import cv2

def extract_frames(video_path: str, output_dir: str, interval: int = 15, max_frames: int = 500):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Extracting frames from '{video_name}' (Total frames: {total_frames}, FPS: {fps:.1f})...")
    print(f"Saving 1 frame every {interval} frames into '{output_dir}'...")

    frame_idx = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frame_filename = f"{video_name}_f{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"[COMPLETE] Extracted and saved {saved_count} frames to {output_dir}")
    print("Next step: Upload these frames to CVAT (https://www.cvat.ai) and annotate fish bounding boxes.")

def main():
    parser = argparse.ArgumentParser(description="Extract video frames for CVAT annotation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="dataset/raw_frames", help="Output directory for frames")
    parser.add_argument("--interval", type=int, default=15, help="Extract every N-th frame")
    parser.add_argument("--max", type=int, default=500, help="Maximum number of frames to extract")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.interval, args.max)

if __name__ == "__main__":
    main()
