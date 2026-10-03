# 🐟 SatoLeo Count — AI Fish & Fingerling Counting System

**SatoLeo Count** is an intelligent, high-precision computer vision and object tracking system built specifically for aquaculture farms, fish hatcheries, and research facilities. It enables farmers to point a smartphone camera, webcam, or video recording at a fish container, bowl, or counting channel to detect, track, and accurately calculate the number of unique fish and fingerlings without duplicate counting.

---

## 📑 Table of Contents
1. [Overview & Project Goal](#-1-overview--project-goal)
2. [System Architecture](#-2-system-architecture)
3. [Core AI Pipeline](#-3-core-ai-pipeline)
4. [Installation & Setup](#-4-installation--setup)
   - [Prerequisites](#prerequisites)
   - [Backend Setup (Python)](#backend-setup-python)
   - [Frontend Setup (React + Vite)](#frontend-setup-react--vite)
5. [Quick Start & Testing](#-5-quick-start--testing)
   - [1. Generate Synthetic Aquaculture Demo Video](#1-generate-synthetic-aquaculture-demo-video)
   - [2. Run Standalone AI Tracking & Counting Test](#2-run-standalone-ai-tracking--counting-test)
   - [3. Start the Full Application (Backend + Frontend)](#3-start-the-full-application-backend--frontend)
6. [Counting Strategies](#-6-counting-strategies)
7. [CVAT Annotation Workflow & Custom Training](#-7-cvat-annotation-workflow--custom-training)
8. [Live Camera Architecture & WebSockets](#-8-live-camera-architecture--websockets)
9. [CLI Tools & Scripts](#-9-cli-tools--scripts)
10. [Troubleshooting & FAQ](#-10-troubleshooting--faq)
11. [Production Deployment](#-11-production-deployment)

---

## 🎯 1. Overview & Project Goal

Counting fish and fingerlings manually is labor-intensive, error-prone, and stressful to the fish. A standard object detection system counts detections on *every* frame, leading to thousands of false duplicate counts because the same fish swims across hundreds of frames.

**SatoLeo Count** solves this by combining:
1. **Ultralytics YOLO**: High-speed deep learning object detection for fish/fingerling bodies in varied water turbidities.
2. **ByteTrack**: Motion association using Kalman filtering and bipartite matching with low/high confidence detection association, ensuring individual fish maintain a persistent **Unique ID**.
3. **Multi-Strategy Counter**: Configurable counting engines (Unique ID with multi-frame confirmation, Virtual Line Crossing, and Polygonal Region-of-Interest).

---

## 🏗️ 2. System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                       SATOLEO COUNT                          │
└──────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [ Live Browser Camera ]          [ Uploaded Fish Video ]
       (Desktop / Phone)               (MP4 / AVI / MOV)
               │                               │
               │ (WebSocket Stream)            │ (HTTP Multipart)
               ▼                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │                 FastAPI Backend Service                  │
    │  - REST Endpoints (/api/count/status, /start, /video)    │
    │  - Realtime WebSocket Handler (/api/count/ws)           │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  AI Processing Pipeline                  │
    │                                                          │
    │  1. OpenCV Frame Decoding                                │
    │  2. YOLO Fish Detector (Custom weights / fallback)       │
    │  3. ByteTrack Object Tracker (Unique ID Association)     │
    │  4. Counting Engine (UniqueId / LineCrossing / ROI)      │
    │  5. Visual Annotation & Trajectory Renderer              │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  React + Vite Frontend                   │
    │  - Realtime Live Video Viewport with Bounding Overlays   │
    │  - Large Hero Telemetry Counter ("247 FISH COUNTED")    │
    │  - Camera Switcher (Front / Rear for mobile phones)     │
    │  - Interactive AI Threshold & Mode Sliders               │
    │  - CSV Session Report Exporter                           │
    └──────────────────────────────────────────────────────────┘
```

---

## 🔬 3. Core AI Pipeline

### How YOLO Works in SatoLeo Count
YOLO (You Only Look Once) is an anchor-free single-stage object detector. When a video frame enters the pipeline:
1. The image is resized to `640x640` and normalized.
2. The convolutional backbone and feature pyramid extract multi-scale spatial representations.
3. The detection head predicts bounding boxes `[x1, y1, x2, y2]` along with confidence scores for each fish candidate.

### How ByteTrack Works
Standard trackers discard low-score detections, losing tracks when fish occlude each other or swim deeper underwater. **ByteTrack** retains both high and low score detections:
- **First Association**: Matches high-confidence detections with existing tracklets using spatial overlap (IoU distance) and Kalman Filter velocity prediction.
- **Second Association**: Matches unmatched tracklets with low-confidence detections (e.g. partially occluded fish).
- Each fish is assigned a persistent integer `track_id` (e.g. `Fish #1`, `Fish #2`).

### Why We Do NOT Simply Sum Detections
If 10 fish swim in a 30 FPS video for 10 seconds (300 frames), a simple detector outputs `10 × 300 = 3,000` detections. SatoLeo Count registers each unique track ID once, yielding the true count of **10 fish**.

---

## ⚙️ 4. Installation & Setup

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: v18.0.0 or higher (v20+ recommended)
- **Git**

### Backend Setup (Python)
1. Open a terminal in the project root directory:
   ```bash
   cd "d:/satoleo fish count"
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

### Frontend Setup (React + Vite)
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install frontend packages:
   ```bash
   npm install
   ```

---

## 🚀 5. Quick Start & Testing

### 1. Generate Synthetic Aquaculture Demo Video
Even before recording your own pond or bowl video, generate a realistic 12-fish simulation video:
```bash
python scripts/generate_demo_video.py --output videos/sample_fish.mp4 --num-fish 12 --duration 10
```

### 2. Run Standalone AI Tracking & Counting Test
Run the AI pipeline directly on the video and generate an annotated output video with ID tags and trajectories:
```bash
python scripts/track.py --source videos/sample_fish.mp4 --output videos/outputs/tracked_fish.mp4
```
**Expected Output:**
```text
=================================================================
 SATOLEO COUNT — AI TRACKING & COUNTING ENGINE 
 Input Video   : videos/sample_fish.mp4
 Output Video  : videos/outputs/tracked_fish.mp4
 Counting Mode : unique_id
=================================================================
Processing 300 frames (800x600 @ 30.0 FPS)...
Frame [25/300] (8.3%) | Count: 12 fish | In-frame: 12 | FPS: 34.2
...
FINAL UNIQUE FISH COUNT: 12
Output Video Saved to : videos/outputs/tracked_fish.mp4
```

### 3. Start the Full Application (Backend + Frontend)

**Terminal 1 — Launch FastAPI Backend:**
```bash
python -m backend.main
```
*API will run at `http://localhost:8000` (Swagger docs available at `http://localhost:8000/docs`).*

**Terminal 2 — Launch React Frontend:**
```bash
cd frontend
npm run dev
```
*Frontend will run at `http://localhost:3000`.*

---

## 📊 6. Counting Strategies

SatoLeo Count includes 3 interchangeable counting strategies implemented in `backend/ai/counter.py`:

| Strategy | Ideal Use Case | Description |
| :--- | :--- | :--- |
| **`unique_id`** (Default) | Bowls, Buckets, Tanks | Counts unique ByteTrack IDs that persist for $\ge N$ consecutive frames (`min_hits`), filtering out transient noise. |
| **`line_crossing`** | Narrow channels, Transfer pipes | Counts fish when their trajectory vector intersects a virtual counting line across the flow direction. |
| **`roi`** | Transfer gates, Specific grids | Counts unique fish entering or residing inside a predefined polygonal bounding region. |

---

## 🏷️ 7. CVAT Annotation Workflow & Custom Training

To train SatoLeo's own high-accuracy model for tiny fingerlings:

### Step 1: Record Videos & Extract Frames
Record videos under diverse aquaculture conditions (varying bowl colors, water clarity, lighting, fingerling densities). Extract 1 frame every 15 frames:
```bash
python scripts/extract_frames.py --video videos/my_tank_recording.mp4 --output dataset/raw_frames --interval 15
```

### Step 2: Annotate in CVAT
1. Go to [CVAT.ai](https://www.cvat.ai) and create a new project called `SatoLeo-Fish`.
2. Add a single class label: `fish`.
3. Create a task and upload the extracted images from `dataset/raw_frames`.
4. Draw tight bounding boxes around all visible fish/fingerlings.
5. In CVAT, click **Export Task Dataset** ➔ Select **YOLO 1.1** format ➔ Download ZIP.

### Step 3: Organize Dataset
Extract the downloaded ZIP into the `dataset/` directory:
```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### Step 4: Train Custom SatoLeo Fish Model
Run the custom training script:
```bash
python scripts/train.py --data dataset/data.yaml --epochs 50 --imgsz 640 --batch 16
```
The script trains with augmentations specifically tuned for fish tanks (rotations, HSV water color shifts, scale changes) and automatically copies the best weights to:
`backend/models/satoleo_fish_best.pt`

When restarted, SatoLeo Count automatically detects the custom weights and switches from **Preview Mode** to **Custom SatoLeo AI** mode.

---

## 📡 8. Live Camera Architecture & WebSockets

### Why WebSockets Instead of Raw HTTP
Standard HTTP polling (`GET /frame`) introduces $150\text{–}300\text{ms}$ latency per frame and severe network overhead. 

SatoLeo Count implements a high-throughput **bidirectional WebSocket stream** (`/api/count/ws`):
1. **Frontend**: Captures frames from `navigator.mediaDevices.getUserMedia` via an offscreen HTML5 canvas at $\approx 15\text{–}20\text{ FPS}$.
2. **Transport**: Encodes frames as compressed JPEG blobs and transmits over a single persistent WebSocket connection.
3. **Backend**: Decodes bytes directly into memory (`cv2.imdecode`), runs inference through ByteTrack, annotates the frame with bounding boxes and ID trails, and sends back JSON telemetry + annotated image.
4. **Result**: Glass-to-glass latency under $45\text{ms}$ on local networks.

---

## 🛠️ 9. CLI Tools & Scripts

| Script | Purpose | Command |
| :--- | :--- | :--- |
| `scripts/predict.py` | Test detection on image or video | `python scripts/predict.py --source image.jpg` |
| `scripts/track.py` | Run tracking & unique counting CLI | `python scripts/track.py --source video.mp4` |
| `scripts/train.py` | Fine-tune YOLO on SatoLeo dataset | `python scripts/train.py --epochs 50` |
| `scripts/extract_frames.py` | Slice video into CVAT training frames | `python scripts/extract_frames.py --video video.mp4` |
| `scripts/generate_demo_video.py` | Generate realistic test simulation | `python scripts/generate_demo_video.py` |

---

## ❓ 10. Troubleshooting & FAQ

### Camera permission denied in browser
- **Cause**: Browser blocked camera access or site is not served on `localhost` / `HTTPS`.
- **Fix**: Allow camera permissions in your browser URL bar icon. On mobile, ensure you access via `https://` or `localhost`.

### Model loading fallback warning
- **Cause**: Custom weights file `backend/models/satoleo_fish_best.pt` not found.
- **Behavior**: System gracefully falls back to `yolov8n.pt` and displays **[Test / Preview Mode]** in the UI. Train custom weights using `python scripts/train.py` to activate full production mode.

### Fish count is higher than actual (Duplicate counts)
- **Fix**: In the UI or `config.py`, increase the **Track Stability Filter** (`min_hits`) to `3` or `4` frames, or slightly increase **Confidence Threshold** to ignore spurious reflections.

---

## 🌐 11. Production Deployment

### Docker Deployment
Create a `Dockerfile` with Python 3.11, PyTorch, and OpenCV dependencies, then expose port `8000`:
```bash
docker build -t satoleo-count .
docker run -p 8000:8000 -p 3000:3000 satoleo-count
```

### Mobile Access in Field
To use a smartphone camera in the hatchery:
1. Run backend on your local server or laptop connected to the same Wi-Fi.
2. Open `http://<YOUR_LOCAL_IP>:3000` on your smartphone browser.
3. Tap **Switch Camera** to toggle to the phone's high-resolution rear camera.
#   a i - c o u n t -  
 