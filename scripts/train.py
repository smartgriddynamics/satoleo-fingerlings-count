"""
SatoLeo Count - Custom Fish Detector Training Script
Fine-tunes an Ultralytics YOLO model on the SatoLeo aquaculture dataset.

Usage:
    python scripts/train.py --data dataset/data.yaml --epochs 50 --imgsz 640 --batch 16
"""
import argparse
import sys
import os
import shutil
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train SatoLeo Fish Detection Model")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained base model (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="", help="Device: cuda device '0' or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train", help="Save directory")
    parser.add_argument("--name", type=str, default="satoleo_fish", help="Experiment name")
    parser.add_argument("--export-backend", action="store_true", default=True, help="Auto-copy best weights to backend/models/")
    args = parser.parse_args()

    # Validate dataset config
    if not os.path.exists(args.data):
        print(f"Error: Dataset configuration file not found at {args.data}")
        print("Please prepare your dataset in YOLO format or see README for instructions.")
        sys.exit(1)

    print("=" * 65)
    print(" SATOLEO COUNT — MODEL TRAINING ")
    print(f" Base Model     : {args.model}")
    print(f" Dataset Config : {args.data}")
    print(f" Epochs         : {args.epochs}")
    print(f" Image Size     : {args.imgsz}")
    print(f" Batch Size     : {args.batch}")
    print("=" * 65)

    # Initialize base model
    model = YOLO(args.model)

    # Train model with augmentations tuned for aquaculture/fish fingerling scenarios:
    # (flips, slight rotations, scale jitter, perspective, HSV variations for varying water clarity)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        # Augmentation settings for fish tanks
        degrees=15.0,        # Fish swim in all orientations
        flipud=0.5,          # Vertical flip
        fliplr=0.5,          # Horizontal flip
        hsv_h=0.015,         # Hue variation for water color
        hsv_s=0.6,           # Saturation variation
        hsv_v=0.4,           # Brightness/reflection variation
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )

    # Validate trained model
    print("\nEvaluating trained model on validation set...")
    val_results = model.val()
    print(f"Validation mAP50-95: {val_results.box.map:.4f}")
    print(f"Validation mAP50:    {val_results.box.map50:.4f}")

    # Auto-copy best weights to backend/models/satoleo_fish_best.pt if requested
    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    if args.export_backend and os.path.exists(best_weights):
        target_dir = os.path.abspath("backend/models")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "satoleo_fish_best.pt")
        shutil.copy2(best_weights, target_path)
        print(f"\n[SUCCESS] Exported best model weights to: {target_path}")
        print("SatoLeo Count backend will now automatically use your custom fish model!")

if __name__ == "__main__":
    main()
