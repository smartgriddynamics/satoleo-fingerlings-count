"""
SatoLeo Count - Counting REST API Endpoints
"""
import os
import shutil
import time
import cv2
import logging
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.ai.config import PipelineConfig
from backend.ai.pipeline import FishCountingPipeline
from backend.api.models import (
    CountStatusResponse,
    StartCountRequest,
    VideoProcessResponse,
    UpdateSettingsRequest,
    TrackItem
)

logger = logging.getLogger("satoleo.api.count")
router = APIRouter(prefix="/api/count", tags=["Counting"])

# Global pipeline instance for live counting sessions
pipeline_config = PipelineConfig()
live_pipeline = FishCountingPipeline(pipeline_config)
session_status = "idle"  # idle, counting, paused, stopped

@router.get("/status", response_model=CountStatusResponse)
async def get_count_status():
    """Returns current live counting state, metrics, and active tracks."""
    stats = live_pipeline.counter.get_stats()
    tracks = [
        TrackItem(
            id=obj["track_id"],
            bbox=obj["bbox"],
            conf=obj["confidence"],
            centroid=obj["centroid"]
        )
        for obj in live_pipeline.tracker.active_tracks
    ]

    return CountStatusResponse(
        count=live_pipeline.counter.get_count(),
        active_tracks=len(live_pipeline.tracker.active_tracks),
        fps=round(live_pipeline.fps, 1),
        status=session_status,
        model_name=live_pipeline.detector.model_name,
        is_custom_model=live_pipeline.detector.is_custom_model,
        counting_mode=live_pipeline.config.counting_mode,
        counter_stats=stats,
        recent_tracks=tracks
    )

@router.post("/start")
async def start_counting(req: StartCountRequest = StartCountRequest()):
    """Starts a new counting session, configuring mode or thresholds."""
    global session_status
    live_pipeline.reset()
    if req.conf_threshold:
        live_pipeline.config.conf_threshold = req.conf_threshold
    if req.mode:
        live_pipeline.config.counting_mode = req.mode
        live_pipeline.counter = None  # Re-init counter
        from backend.ai.counter import create_counter
        live_pipeline.counter = create_counter(live_pipeline.config)
    if req.min_hits:
        live_pipeline.config.min_hits = req.min_hits

    session_status = "counting"
    logger.info("Counting session started.")
    return {
        "status": "counting",
        "message": "Fish counting session initialized successfully.",
        "config": {
            "mode": live_pipeline.config.counting_mode,
            "conf_threshold": live_pipeline.config.conf_threshold,
            "min_hits": live_pipeline.config.min_hits,
            "model": live_pipeline.detector.model_name
        }
    }

@router.post("/stop")
async def stop_counting():
    """Stops the current counting session and returns summary metrics."""
    global session_status
    session_status = "stopped"
    count = live_pipeline.counter.get_count()
    stats = live_pipeline.counter.get_stats()
    logger.info(f"Counting session stopped. Final count: {count}")
    return {
        "status": "stopped",
        "final_count": count,
        "stats": stats
    }

@router.post("/reset")
async def reset_counter():
    """Resets the fish count and tracking history back to 0."""
    global session_status
    live_pipeline.reset()
    session_status = "idle"
    logger.info("Counter and tracker reset to zero.")
    return {"status": "idle", "count": 0, "message": "Counter reset successfully."}

@router.post("/settings")
async def update_settings(req: UpdateSettingsRequest):
    """Dynamically updates confidence threshold, counting mode, or tracking buffer."""
    if req.conf_threshold is not None:
        live_pipeline.config.conf_threshold = req.conf_threshold
    if req.iou_threshold is not None:
        live_pipeline.config.iou_threshold = req.iou_threshold
    if req.counting_mode is not None:
        live_pipeline.config.counting_mode = req.counting_mode
        from backend.ai.counter import create_counter
        live_pipeline.counter = create_counter(live_pipeline.config)
    if req.min_hits is not None:
        live_pipeline.config.min_hits = req.min_hits
    if req.track_buffer is not None:
        live_pipeline.config.track_buffer = req.track_buffer

    return {
        "message": "Settings updated successfully.",
        "current_config": {
            "conf_threshold": live_pipeline.config.conf_threshold,
            "iou_threshold": live_pipeline.config.iou_threshold,
            "counting_mode": live_pipeline.config.counting_mode,
            "min_hits": live_pipeline.config.min_hits
        }
    }

@router.post("/video", response_model=VideoProcessResponse)
async def process_video_file(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    mode: str = Form("unique_id"),
    min_hits: int = Form(2)
):
    """
    Processes an uploaded video file frame-by-frame with OpenCV + YOLO + ByteTrack.
    Saves the annotated video and returns structured counts and performance metrics.
    """
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format. Please upload MP4, AVI, or MOV.")

    upload_dir = os.path.abspath("videos/uploads")
    output_dir = os.path.abspath("videos/outputs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    safe_name = f"{timestamp}_{file.filename}"
    input_path = os.path.join(upload_dir, safe_name)
    output_filename = f"annotated_{safe_name}"
    output_path = os.path.join(output_dir, output_filename)

    # Save uploaded video
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create dedicated video processing pipeline
    v_config = PipelineConfig(
        conf_threshold=conf,
        counting_mode=mode,
        min_hits=min_hits
    )
    v_pipeline = FishCountingPipeline(v_config)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open uploaded video with OpenCV.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            annotated, _ = v_pipeline.process_frame(frame, annotate=True)
            out.write(annotated)
    finally:
        cap.release()
        out.release()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    final_count = v_pipeline.counter.get_count()
    counter_stats = v_pipeline.counter.get_stats()

    return VideoProcessResponse(
        status="completed",
        filename=file.filename,
        total_frames=frame_count,
        final_count=final_count,
        avg_fps=round(avg_fps, 1),
        elapsed_time=round(elapsed, 2),
        output_video_url=f"/outputs/{output_filename}",
        is_custom_model=v_pipeline.detector.is_custom_model,
        model_name=v_pipeline.detector.model_name,
        counter_stats=counter_stats
    )
