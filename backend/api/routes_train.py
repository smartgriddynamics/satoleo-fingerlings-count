"""
SatoLeo Count - Custom AI Model Training & Dataset Collector API
Allows users to annotate custom species/eggs/fingerlings, collect datasets,
run fine-tuning training jobs, and switch active detection models live.
"""
import os
import json
import time
import shutil
import logging
import threading
import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from backend.api.routes_count import live_pipeline, pipeline_config

logger = logging.getLogger("satoleo.api.train")
router = APIRouter(prefix="/api/train", tags=["Custom AI Studio"])

# Paths for custom dataset and trained models
CUSTOM_DATASET_DIR = os.path.abspath("dataset/custom")
IMAGES_DIR = os.path.join(CUSTOM_DATASET_DIR, "images")
LABELS_DIR = os.path.join(CUSTOM_DATASET_DIR, "labels")
YAML_PATH = os.path.join(CUSTOM_DATASET_DIR, "data.yaml")
CLASSES_JSON_PATH = os.path.join(CUSTOM_DATASET_DIR, "classes.json")
MODELS_DIR = os.path.abspath("backend/models")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training Job State
training_state = {
    "status": "idle",  # idle, training, completed, failed
    "progress_pct": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "map50": 0.0,
    "message": "Ready to collect samples and train custom AI models.",
    "model_name": "",
    "logs": []
}

class BoundingBox(BaseModel):
    x: float  # Normalized center_x (0..1) or pixel x
    y: float  # Normalized center_y (0..1) or pixel y
    w: float  # Normalized width (0..1) or pixel width
    h: float  # Normalized height (0..1) or pixel height
    label: str  # Class name (e.g. "Fish Egg", "Tilapia")

class SaveSampleRequest(BaseModel):
    image_b64: str
    boxes: List[BoundingBox]

class StartTrainRequest(BaseModel):
    model_name: str = "custom_fish_model"
    epochs: int = 15
    batch_size: int = 8
    base_model: str = "yolov8n.pt"

class AddClassRequest(BaseModel):
    class_name: str

class SelectModelRequest(BaseModel):
    model_filename: str

def _get_classes() -> List[str]:
    if os.path.exists(CLASSES_JSON_PATH):
        try:
            with open(CLASSES_JSON_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return ["Fish", "Fish Egg", "Fingerling"]

def _save_classes(classes: List[str]):
    with open(CLASSES_JSON_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    _update_data_yaml(classes)

def _update_data_yaml(classes: List[str]):
    yaml_content = f"""path: {CUSTOM_DATASET_DIR}
train: images
val: images

names:
"""
    for idx, cls_name in enumerate(classes):
        yaml_content += f"  {idx}: '{cls_name}'\n"

    with open(YAML_PATH, "w") as f:
        f.write(yaml_content)

# Initialize classes if not present
if not os.path.exists(CLASSES_JSON_PATH):
    _save_classes(["Fish", "Fish Egg", "Fingerling"])

@router.get("/classes")
async def get_classes():
    """Returns all registered custom object/species classes."""
    return {"classes": _get_classes()}

@router.post("/classes")
async def add_class(req: AddClassRequest):
    """Adds a new custom species/object class to the taxonomy."""
    cls_name = req.class_name.strip()
    if not cls_name:
        raise HTTPException(status_code=400, detail="Class name cannot be empty")
    
    classes = _get_classes()
    if cls_name not in classes:
        classes.append(cls_name)
        _save_classes(classes)
    
    return {"classes": classes, "message": f"Class '{cls_name}' registered."}

@router.get("/dataset/stats")
async def get_dataset_stats():
    """Returns dataset sample count and annotation statistics per class."""
    classes = _get_classes()
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    label_counts = {cls: 0 for cls in classes}

    for label_file in os.listdir(LABELS_DIR):
        if label_file.endswith('.txt'):
            path = os.path.join(LABELS_DIR, label_file)
            try:
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_idx = int(parts[0])
                            if 0 <= cls_idx < len(classes):
                                label_counts[classes[cls_idx]] += 1
            except Exception:
                pass

    return {
        "total_samples": len(image_files),
        "classes": classes,
        "annotations_per_class": label_counts
    }

@router.post("/dataset/sample")
async def save_dataset_sample(req: SaveSampleRequest):
    """Saves a user snapshot and YOLO formatted bounding box annotations."""
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="Image data is required")
    
    classes = _get_classes()

    # Decode base64 image
    img_b64 = req.image_b64
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    
    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = frame.shape[:2]
    timestamp = int(time.time() * 1000)
    img_filename = f"sample_{timestamp}.jpg"
    txt_filename = f"sample_{timestamp}.txt"

    img_path = os.path.join(IMAGES_DIR, img_filename)
    txt_path = os.path.join(LABELS_DIR, txt_filename)

    # Save image file
    cv2.imwrite(img_path, frame)

    # Format YOLO lines: <class_id> <x_center> <y_center> <width> <height> (all normalized 0..1)
    yolo_lines = []
    for box in req.boxes:
        label = box.label
        if label not in classes:
            classes.append(label)
            _save_classes(classes)
        
        cls_idx = classes.index(label)

        # Normalize coordinates if they were sent in pixels
        x_center = box.x / w if box.x > 1.0 else box.x
        y_center = box.y / h if box.y > 1.0 else box.y
        box_w = box.w / w if box.w > 1.0 else box.w
        box_h = box.h / h if box.h > 1.0 else box.h

        # Clamp values between 0 and 1
        x_center = max(0.001, min(0.999, x_center))
        y_center = max(0.001, min(0.999, y_center))
        box_w = max(0.001, min(0.999, box_w))
        box_h = max(0.001, min(0.999, box_h))

        yolo_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return {
        "status": "success",
        "message": f"Saved sample {img_filename} with {len(req.boxes)} annotations.",
        "sample_id": timestamp
    }

@router.post("/dataset/clear")
async def clear_custom_dataset():
    """Clears collected custom training samples."""
    for folder in [IMAGES_DIR, LABELS_DIR]:
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return {"message": "Custom dataset cleared successfully."}

def _run_training_job(req: StartTrainRequest):
    """Background worker function for Ultralytics YOLO training."""
    global training_state
    try:
        training_state["status"] = "training"
        training_state["progress_pct"] = 5
        training_state["total_epochs"] = req.epochs
        training_state["current_epoch"] = 0
        training_state["message"] = f"Initializing training for model '{req.model_name}'..."
        training_state["logs"] = [f"Starting YOLO fine-tuning on custom dataset for {req.epochs} epochs."]

        from ultralytics import YOLO

        # Check dataset config
        _update_data_yaml(_get_classes())

        model = YOLO(req.base_model)
        
        output_name = f"custom_{req.model_name.replace(' ', '_').lower()}"
        save_dir = os.path.abspath("runs/custom_train")

        def epoch_callback(trainer):
            ep = trainer.epoch + 1
            tot = trainer.epochs
            pct = int((ep / tot) * 90) + 5
            training_state["current_epoch"] = ep
            training_state["progress_pct"] = pct
            training_state["message"] = f"Training Epoch {ep}/{tot} completed."
            training_state["logs"].append(f"Epoch {ep}/{tot} complete. Loss: {round(float(getattr(trainer, 'loss', 0.0) or 0.0), 4)}")

        model.add_callback("on_train_epoch_end", epoch_callback)

        results = model.train(
            data=YAML_PATH,
            epochs=req.epochs,
            imgsz=640,
            batch=req.batch_size,
            project=save_dir,
            name=output_name,
            degrees=15.0,
            flipud=0.5,
            fliplr=0.5,
            save=True,
            verbose=False
        )

        best_pt = os.path.join(save_dir, output_name, "weights", "best.pt")
        target_filename = f"{output_name}.pt"
        target_pt = os.path.join(MODELS_DIR, target_filename)

        if os.path.exists(best_pt):
            shutil.copy2(best_pt, target_pt)
            shutil.copy2(best_pt, os.path.join(MODELS_DIR, "satoleo_fish_best.pt"))

        training_state["status"] = "completed"
        training_state["progress_pct"] = 100
        training_state["model_name"] = target_filename
        training_state["message"] = f"Training complete! Model saved as '{target_filename}'."
        training_state["logs"].append(f"SUCCESS: Saved custom fine-tuned weights to backend/models/{target_filename}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_state["status"] = "failed"
        training_state["message"] = f"Training error: {str(e)}"
        training_state["logs"].append(f"ERROR: {str(e)}")

@router.post("/start")
async def start_training(req: StartTrainRequest, background_tasks: BackgroundTasks):
    """Starts custom AI fine-tuning job in background."""
    global training_state
    if training_state["status"] == "training":
        raise HTTPException(status_code=400, detail="A training job is already in progress.")
    
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(image_files) < 1:
        raise HTTPException(status_code=400, detail="Please collect and annotate at least 1 image before training!")

    thread = threading.Thread(target=_run_training_job, args=(req,), daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": f"Training job '{req.model_name}' initiated for {req.epochs} epochs.",
        "dataset_samples": len(image_files)
    }

@router.get("/status")
async def get_training_status():
    """Returns real-time status and logs of the AI training process."""
    return training_state

@router.get("/models")
async def list_available_models():
    """Lists all available AI models (base models and custom trained models)."""
    models = ["yolov8n.pt"]
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pt") and f not in models:
                models.append(f)

    active_model = live_pipeline.detector.model_name
    return {
        "models": models,
        "active_model": active_model,
        "is_custom": live_pipeline.detector.is_custom_model
    }

@router.post("/select-model")
async def select_active_model(req: SelectModelRequest):
    """Instantly switches the active detector model in the live AI stream."""
    model_name = req.model_filename.strip()
    
    if model_name == "yolov8n.pt":
        target_path = "yolov8n.pt"
    else:
        target_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail=f"Model file '{model_name}' not found in backend/models/")

    live_pipeline.config.model_path = target_path
    live_pipeline.detector._load_model()
    
    return {
        "status": "success",
        "active_model": live_pipeline.detector.model_name,
        "is_custom": live_pipeline.detector.is_custom_model,
        "message": f"Switched live AI detector model to '{live_pipeline.detector.model_name}'."
    }
