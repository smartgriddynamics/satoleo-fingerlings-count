"""
SatoLeo Count - Realtime WebSocket Stream Handler
Enables ultra low-latency live camera streaming from browser -> backend AI -> frontend.
"""
import base64
import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.api.routes_count import live_pipeline

logger = logging.getLogger("satoleo.api.stream")
router = APIRouter(prefix="/api/count", tags=["Live Stream"])

@router.websocket("/ws")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live browser camera frame processing.
    Accepts JSON messages:
    - {"type": "frame", "image": "data:image/jpeg;base64,..."}
    - {"type": "command", "action": "reset" | "set_conf" | "set_mode", ...}
    """
    await websocket.accept()
    logger.info("WebSocket client connected for live camera counting.")

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            msg_type = data.get("type", "frame")

            if msg_type == "command":
                action = data.get("action")
                if action == "reset":
                    live_pipeline.reset()
                    await websocket.send_json({"type": "status", "message": "Counter reset", "count": 0})
                elif action == "set_conf":
                    conf = float(data.get("value", 0.25))
                    live_pipeline.config.conf_threshold = conf
                    await websocket.send_json({"type": "status", "message": f"Confidence set to {conf}"})
                elif action == "set_mode":
                    mode = data.get("value", "unique_id")
                    live_pipeline.config.counting_mode = mode
                    from backend.ai.counter import create_counter
                    live_pipeline.counter = create_counter(live_pipeline.config)
                    await websocket.send_json({"type": "status", "message": f"Mode set to {mode}"})
                continue

            # Process Video Frame
            img_b64 = data.get("image", "")
            if not img_b64:
                continue

            # Strip data URL header if present
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            # Decode image from base64
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"type": "error", "message": "Failed to decode image frame"})
                continue

            # Process through SatoLeo AI pipeline
            annotated_frame, metrics = live_pipeline.process_frame(frame, annotate=True)

            # Encode annotated frame back to JPEG for live visualization overlay
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            # Send back live metrics and annotated frame
            response_payload = {
                "type": "frame_result",
                "count": metrics["count"],
                "active_tracks": metrics["active_tracks"],
                "fps": metrics["fps"],
                "frame_number": metrics["frame_number"],
                "is_custom_model": metrics["is_custom_model"],
                "model_name": metrics["model_name"],
                "tracks": metrics["tracks"],
                "annotated_frame": annotated_b64
            }

            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
