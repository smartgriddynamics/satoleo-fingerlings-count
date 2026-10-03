import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Fish,
  Camera,
  Upload,
  Play,
  Pause,
  RotateCcw,
  Download,
  Settings2,
  Activity,
  Zap,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Video,
  ShieldCheck,
  Radio,
  Sliders,
  Maximize2,
  Plus,
  Trash2,
  Cpu,
  Layers,
  Sparkles,
  Save,
  Target,
  Crosshair
} from 'lucide-react';

export default function App() {
  // Mode selection: 'camera' | 'upload' | 'studio'
  const [activeTab, setActiveTab] = useState('camera');

  // Live Stream / Camera state
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [facingMode, setFacingMode] = useState('environment'); // 'user' or 'environment' for phone back camera
  const [cameraError, setCameraError] = useState(null);

  // AI Telemetry
  const [count, setCount] = useState(0);
  const [activeTracks, setActiveTracks] = useState(0);
  const [fps, setFps] = useState(0);
  const [modelName, setModelName] = useState('yolov8n.pt');
  const [isCustomModel, setIsCustomModel] = useState(false);
  const [recentTracks, setRecentTracks] = useState([]);
  const [annotatedFrameUrl, setAnnotatedFrameUrl] = useState(null);

  // Configuration parameters
  const [countingMode, setCountingMode] = useState('unique_id'); // unique_id, line_crossing, roi
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [minHits, setMinHits] = useState(2);

  // Upload video state
  const [uploadFile, setUploadFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Custom AI Studio State
  const [studioImageSrc, setStudioImageSrc] = useState(null);
  const [drawnBoxes, setDrawnBoxes] = useState([]);
  const [currentLabel, setCurrentLabel] = useState('Fish Egg');
  const [customClasses, setCustomClasses] = useState(['Fish', 'Fish Egg', 'Fingerling']);
  const [newClassName, setNewClassName] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [liveDrawBox, setLiveDrawBox] = useState(null);

  // Dataset & Training State
  const [datasetStats, setDatasetStats] = useState({ total_samples: 0, classes: [], annotations_per_class: {} });
  const [trainModelName, setTrainModelName] = useState('fish_eggs_v1');
  const [trainEpochs, setTrainEpochs] = useState(15);
  const [trainStatus, setTrainStatus] = useState({ status: 'idle', progress_pct: 0, message: '', logs: [] });
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModelToLoad, setSelectedModelToLoad] = useState('yolov8n.pt');
  const [studioMessage, setStudioMessage] = useState(null);

  // DOM Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamIntervalRef = useRef(null);
  const fileInputRef = useRef(null);
  const studioCanvasRef = useRef(null);
  const studioFileInputRef = useRef(null);
  const trainPollRef = useRef(null);

  // Fetch initial system status
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/count/status');
      if (res.ok) {
        const data = await res.json();
        setCount(data.count);
        setActiveTracks(data.active_tracks);
        setFps(data.fps);
        setModelName(data.model_name);
        setIsCustomModel(data.is_custom_model);
        setCountingMode(data.counting_mode || 'unique_id');
      }
    } catch (err) {
      console.warn('Backend server not connected yet:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  // Studio API Handlers
  const fetchClasses = async () => {
    try {
      const res = await fetch('/api/train/classes');
      if (res.ok) {
        const data = await res.json();
        setCustomClasses(data.classes);
        if (data.classes.length > 0 && !data.classes.includes(currentLabel)) {
          setCurrentLabel(data.classes[0]);
        }
      }
    } catch (e) {
      console.error('Failed to fetch classes:', e);
    }
  };

  const fetchDatasetStats = async () => {
    try {
      const res = await fetch('/api/train/dataset/stats');
      if (res.ok) {
        const data = await res.json();
        setDatasetStats(data);
      }
    } catch (e) {
      console.error('Failed to fetch dataset stats:', e);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const res = await fetch('/api/train/models');
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data.models);
        setSelectedModelToLoad(data.active_model);
      }
    } catch (e) {
      console.error('Failed to fetch available models:', e);
    }
  };

  const handleAddClass = async () => {
    if (!newClassName.trim()) return;
    try {
      const res = await fetch('/api/train/classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ class_name: newClassName.trim() })
      });
      if (res.ok) {
        const data = await res.json();
        setCustomClasses(data.classes);
        setCurrentLabel(newClassName.trim());
        setNewClassName('');
        setStudioMessage({ type: 'success', text: `Class '${newClassName.trim()}' registered!` });
      }
    } catch (e) {
      console.error('Failed to add class:', e);
    }
  };

  const handleSaveSample = async () => {
    if (!studioImageSrc) {
      alert('Please snap a camera frame or upload an image first!');
      return;
    }
    if (drawnBoxes.length === 0) {
      alert('Please draw at least one bounding box over a fish or egg to annotate!');
      return;
    }

    try {
      const res = await fetch('/api/train/dataset/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: studioImageSrc,
          boxes: drawnBoxes
        })
      });

      if (res.ok) {
        const data = await res.json();
        setStudioMessage({ type: 'success', text: data.message });
        setDrawnBoxes([]);
        fetchDatasetStats();
      } else {
        const errData = await res.json();
        alert(`Failed to save sample: ${errData.detail}`);
      }
    } catch (e) {
      alert(`Save error: ${e.message}`);
    }
  };

  const handleClearDataset = async () => {
    if (!window.confirm('Are you sure you want to clear all collected training samples?')) return;
    try {
      await fetch('/api/train/dataset/clear', { method: 'POST' });
      fetchDatasetStats();
      setStudioMessage({ type: 'info', text: 'Training dataset cleared.' });
    } catch (e) {
      console.error('Clear dataset error:', e);
    }
  };

  const handleStartTraining = async () => {
    if (datasetStats.total_samples < 1) {
      alert('Please annotate and save at least 1 image sample before starting AI training!');
      return;
    }

    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: trainModelName,
          epochs: parseInt(trainEpochs),
          batch_size: 8,
          base_model: 'yolov8n.pt'
        })
      });

      if (res.ok) {
        setTrainStatus({ status: 'training', progress_pct: 5, message: 'Initializing training...', logs: [] });
        startTrainStatusPolling();
      } else {
        const err = await res.json();
        alert(`Training error: ${err.detail}`);
      }
    } catch (e) {
      alert(`Failed to start training: ${e.message}`);
    }
  };

  const startTrainStatusPolling = () => {
    if (trainPollRef.current) clearInterval(trainPollRef.current);
    trainPollRef.current = setInterval(async () => {
      try {
        const res = await fetch('/api/train/status');
        if (res.ok) {
          const data = await res.json();
          setTrainStatus(data);
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(trainPollRef.current);
            fetchAvailableModels();
            fetchStatus();
          }
        }
      } catch (e) {
        console.error('Train status poll error:', e);
      }
    }, 1500);
  };

  const handleSelectModel = async (modelFileName) => {
    try {
      const res = await fetch('/api/train/select-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_filename: modelFileName })
      });

      if (res.ok) {
        const data = await res.json();
        setModelName(data.active_model);
        setIsCustomModel(data.is_custom);
        setStudioMessage({ type: 'success', text: `Active model switched to ${data.active_model}` });
        fetchStatus();
      }
    } catch (e) {
      alert(`Model selection error: ${e.message}`);
    }
  };

  // Canvas Drawing Handlers
  const handleSnapFromCamera = () => {
    if (!videoRef.current) {
      alert('Camera is offline. Please start camera or upload an image!');
      return;
    }
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    setStudioImageSrc(dataUrl);
    setDrawnBoxes([]);
  };

  const handleStudioImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      setStudioImageSrc(evt.target.result);
      setDrawnBoxes([]);
    };
    reader.readAsDataURL(file);
  };

  // Redraw studio canvas whenever image or boxes update
  useEffect(() => {
    if (!studioCanvasRef.current || !studioImageSrc) return;
    const canvas = studioCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = studioImageSrc;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Draw saved boxes
      drawnBoxes.forEach((box, idx) => {
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        ctx.fillStyle = 'rgba(6, 182, 212, 0.25)';
        ctx.fillRect(box.x, box.y, box.w, box.h);

        // Label tag
        ctx.fillStyle = '#0284c7';
        ctx.font = 'bold 14px sans-serif';
        const labelText = `#${idx + 1} ${box.label}`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(box.x, Math.max(0, box.y - 22), textWidth + 10, 22);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, box.x + 5, Math.max(15, box.y - 6));
      });

      // Draw live drawing box
      if (liveDrawBox) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(liveDrawBox.x, liveDrawBox.y, liveDrawBox.w, liveDrawBox.h);
        ctx.setLineDash([]);
      }
    };
  }, [studioImageSrc, drawnBoxes, liveDrawBox]);

  const handleCanvasMouseDown = (e) => {
    if (!studioCanvasRef.current) return;
    const rect = studioCanvasRef.current.getBoundingClientRect();
    const scaleX = studioCanvasRef.current.width / rect.width;
    const scaleY = studioCanvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setIsDrawing(true);
    setDrawStart({ x, y });
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDrawing || !drawStart || !studioCanvasRef.current) return;
    const rect = studioCanvasRef.current.getBoundingClientRect();
    const scaleX = studioCanvasRef.current.width / rect.width;
    const scaleY = studioCanvasRef.current.height / rect.height;
    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;

    setLiveDrawBox({
      x: Math.min(drawStart.x, currentX),
      y: Math.min(drawStart.y, currentY),
      w: Math.abs(currentX - drawStart.x),
      h: Math.abs(currentY - drawStart.y)
    });
  };

  const handleCanvasMouseUp = () => {
    if (isDrawing && liveDrawBox && liveDrawBox.w > 8 && liveDrawBox.h > 8) {
      setDrawnBoxes([...drawnBoxes, { ...liveDrawBox, label: currentLabel }]);
    }
    setIsDrawing(false);
    setDrawStart(null);
    setLiveDrawBox(null);
  };

  // Initialize WebSocket connection for low-latency video streaming
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/count/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('SatoLeo AI WebSocket connected.');
      setCameraError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'frame_result') {
          setCount(data.count);
          setActiveTracks(data.active_tracks);
          setFps(data.fps);
          setIsCustomModel(data.is_custom_model);
          setModelName(data.model_name);
          if (data.tracks) setRecentTracks(data.tracks);
          if (data.annotated_frame) {
            setAnnotatedFrameUrl(data.annotated_frame);
          }
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    ws.onerror = (err) => {
      console.error('WS Error:', err);
    };

    ws.onclose = () => {
      console.log('WS Connection closed.');
    };

    wsRef.current = ws;
    return ws;
  }, []);

  // Capture frame from browser camera and send to backend
  const sendFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || isPaused) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.readyState >= 2 && video.videoWidth > 0) {
      canvas.width = 640;
      canvas.height = (640 / video.videoWidth) * video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const base64Img = canvas.toDataURL('image/jpeg', 0.65);

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'frame', image: base64Img }));
      }
    }
  }, [isPaused]);

  // Start Camera Stream
  const startCamera = async () => {
    setCameraError(null);
    try {
      const constraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        console.warn('Constrained camera getUserMedia failed, retrying basic video constraint:', err);
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      setIsStreaming(true);
      setIsPaused(false);
      connectWebSocket();

      const attachAndPlay = async () => {
        if (videoRef.current) {
          if (videoRef.current.srcObject !== stream) {
            videoRef.current.srcObject = stream;
          }
          try {
            await videoRef.current.play();
          } catch (e) {
            console.warn('Video play prevented:', e);
          }
        }
      };

      await attachAndPlay();
      setTimeout(attachAndPlay, 80);

      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = setInterval(sendFrame, 65);
    } catch (err) {
      console.error('Camera access failed:', err);
      setCameraError(`Camera Error: ${err.message || 'Could not access device camera. Check browser permissions.'}`);
      setIsStreaming(false);
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
    setIsPaused(false);
    setAnnotatedFrameUrl(null);
  };

  // Switch between front/back camera
  const toggleFacingMode = async () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    if (isStreaming) {
      stopCamera();
      setTimeout(startCamera, 300);
    }
  };

  // Reset Counter
  const handleReset = async () => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'command', action: 'reset' }));
      } else {
        await fetch('/api/count/reset', { method: 'POST' });
      }
      setCount(0);
      setActiveTracks(0);
      setRecentTracks([]);
    } catch (err) {
      console.error('Reset failed:', err);
    }
  };

  // Update Counting Settings
  const handleSettingsChange = async (newConf, newMode, newHits) => {
    try {
      await fetch('/api/count/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conf_threshold: newConf !== undefined ? newConf : confThreshold,
          counting_mode: newMode !== undefined ? newMode : countingMode,
          min_hits: newHits !== undefined ? newHits : minHits
        })
      });
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        if (newConf !== undefined) wsRef.current.send(JSON.stringify({ type: 'command', action: 'set_conf', value: newConf }));
        if (newMode !== undefined) wsRef.current.send(JSON.stringify({ type: 'command', action: 'set_mode', value: newMode }));
      }
    } catch (err) {
      console.error('Failed to update settings:', err);
    }
  };

  // Video File Upload Analysis
  const handleVideoUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadFile(file);
    setIsUploading(true);
    setUploadResult(null);
    setUploadProgress(15);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', confThreshold.toString());
    formData.append('mode', countingMode);
    formData.append('min_hits', minHits.toString());

    try {
      const res = await fetch('/api/count/video', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        throw new Error(`Upload analysis failed with status: ${res.status}`);
      }

      const data = await res.json();
      setUploadResult(data);
      setCount(data.final_count);
    } catch (err) {
      alert(`Video processing error: ${err.message}`);
    } finally {
      setIsUploading(false);
      setUploadProgress(100);
    }
  };

  // Save Session Report (CSV export)
  const exportSessionReport = () => {
    const timestamp = new Date().toISOString().replace(/:/g, '-').slice(0, 19);
    const csvContent = "data:text/csv;charset=utf-8," 
      + "SatoLeo Count - Aquaculture Session Report\n"
      + `Timestamp,${new Date().toLocaleString()}\n`
      + `Total Unique Fish Counted,${count}\n`
      + `Current Active in Frame,${activeTracks}\n`
      + `Counting Strategy,${countingMode}\n`
      + `Confidence Threshold,${confThreshold}\n`
      + `AI Model,${modelName}\n`
      + `Custom SatoLeo Weights,${isCustomModel ? 'Yes' : 'No (Test Mode)'}\n`;

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `satoleo_fish_count_${timestamp}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="brand-wrapper">
            <div className="brand-icon-box">
              <Fish size={24} />
            </div>
            <div>
              <h1 className="brand-title">
                SATOLEO <span style={{ color: 'var(--primary)', fontWeight: 400 }}>COUNT</span>
              </h1>
              <span className="brand-subtitle">Aquaculture AI Vision System</span>
            </div>
          </div>

          <div className="status-badges">
            {isCustomModel ? (
              <span className="badge badge-green">
                <ShieldCheck size={14} /> Custom SatoLeo AI ({modelName})
              </span>
            ) : (
              <span className="badge badge-amber" title="Trained weights not loaded yet. Using standard YOLO model.">
                <AlertTriangle size={14} /> Test / Base Model ({modelName})
              </span>
            )}

            <span className="badge badge-blue">
              <span className="pulse-dot" /> {isStreaming ? 'Live Stream Active' : 'System Ready'}
            </span>
          </div>
        </div>
      </header>

      {/* Hidden canvas used for frame capture */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Main Dashboard Layout */}
      <main className="main-dashboard">
        {/* Left Column: Video Viewport & Controls */}
        <section className="video-card">
          <div className="card-header-tabs">
            <div className="tab-group">
              <button
                className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab('camera');
                  if (uploadResult) setUploadResult(null);
                }}
              >
                <Camera size={16} /> Live Camera
              </button>
              <button
                className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => {
                  if (isStreaming) stopCamera();
                  setActiveTab('upload');
                }}
              >
                <Upload size={16} /> Upload Video
              </button>
              <button
                className={`tab-btn ${activeTab === 'studio' ? 'active' : ''}`}
                onClick={() => {
                  if (isStreaming) stopCamera();
                  setActiveTab('studio');
                  fetchDatasetStats();
                  fetchClasses();
                  fetchAvailableModels();
                }}
              >
                <Sparkles size={16} /> Custom AI Studio (Train)
              </button>
            </div>

            {activeTab === 'camera' && (
              <button
                className="btn btn-secondary"
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.775rem' }}
                onClick={toggleFacingMode}
                title="Switch between front and back camera on mobile"
              >
                <RefreshCw size={13} /> {facingMode === 'environment' ? 'Rear Camera' : 'Front Camera'}
              </button>
            )}
          </div>

          {/* Viewport Area */}
          <div className="video-viewport">
            {activeTab === 'camera' ? (
              <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="viewport-video-el"
                  style={
                    !isStreaming
                      ? { display: 'none' }
                      : annotatedFrameUrl
                      ? { position: 'absolute', opacity: 0, pointerEvents: 'none', width: '1px', height: '1px' }
                      : {}
                  }
                />
                {isStreaming && annotatedFrameUrl && (
                  <img
                    src={annotatedFrameUrl}
                    alt="SatoLeo Live AI Feed"
                    className="viewport-stream-img"
                  />
                )}
                {!isStreaming && (
                  <div className="camera-placeholder">
                    <div className="placeholder-icon">
                      <Camera size={36} />
                    </div>
                    <h3 style={{ color: '#f8fafc', marginBottom: '0.5rem', fontWeight: 700 }}>
                      Camera Offline
                    </h3>
                    <p style={{ maxWidth: '400px', fontSize: '0.875rem', marginBottom: '1.5rem', color: '#94a3b8' }}>
                      Point your camera directly at the fish container, bowl, or counting channel with good lighting.
                    </p>
                    <button className="btn btn-primary" onClick={startCamera}>
                      <Play size={16} /> Start Live Camera
                    </button>
                    {cameraError && (
                      <div style={{ marginTop: '1rem', color: '#f87171', fontSize: '0.8rem', background: 'rgba(239, 68, 68, 0.1)', padding: '0.5rem 1rem', borderRadius: '8px' }}>
                        {cameraError}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : activeTab === 'upload' ? (
              /* Video Upload Area */
              <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '440px' }}>
                {uploadResult ? (
                  <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <video
                      controls
                      autoPlay
                      loop
                      src={uploadResult.output_video_url}
                      className="viewport-video-el"
                    />
                  </div>
                ) : isUploading ? (
                  <div className="camera-placeholder">
                    <div className="placeholder-icon" style={{ animation: 'spin 2s linear infinite' }}>
                      <RefreshCw size={36} />
                    </div>
                    <h3 style={{ color: '#f8fafc', marginBottom: '0.5rem', fontWeight: 700 }}>
                      Processing Fish Video...
                    </h3>
                    <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                      Running YOLO detection, ByteTrack tracking, and unique ID counting.
                    </p>
                  </div>
                ) : (
                  <div className="camera-placeholder">
                    <div className="placeholder-icon" style={{ color: '#06b6d4' }}>
                      <Upload size={36} />
                    </div>
                    <h3 style={{ color: '#f8fafc', marginBottom: '0.5rem', fontWeight: 700 }}>
                      Upload Video File
                    </h3>
                    <p style={{ maxWidth: '400px', fontSize: '0.875rem', marginBottom: '1.5rem', color: '#94a3b8' }}>
                      Select MP4, MOV, or AVI video of fish swimming in tanks or counting pipes.
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*"
                      style={{ display: 'none' }}
                      onChange={handleVideoUpload}
                    />
                    <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
                      <Upload size={16} /> Choose Video File
                    </button>
                  </div>
                )}
              </div>
            ) : (
              /* Custom Model AI Studio Area */
              <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', padding: '1.25rem', gap: '1rem', color: '#e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255, 255, 255, 0.03)', padding: '0.75rem 1rem', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                  <div>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0284c7', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Target size={18} /> Interactive AI Dataset Annotator
                    </h3>
                    <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                      Snap camera frames or upload photos, click & drag boxes over fish or fish eggs, and fine-tune your custom model!
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }} onClick={handleSnapFromCamera}>
                      <Camera size={14} /> Snap Live Camera Frame
                    </button>
                    <input
                      ref={studioFileInputRef}
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      onChange={handleStudioImageUpload}
                    />
                    <button className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }} onClick={() => studioFileInputRef.current?.click()}>
                      <Upload size={14} /> Upload Photo
                    </button>
                  </div>
                </div>

                {/* Annotation Canvas Stage */}
                <div style={{ position: 'relative', width: '100%', minHeight: '380px', background: '#030712', borderRadius: '12px', border: '1px border var(--border)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {studioImageSrc ? (
                    <div style={{ position: 'relative', cursor: 'crosshair', maxWidth: '100%', maxHeight: '420px', overflow: 'auto' }}>
                      <canvas
                        ref={studioCanvasRef}
                        onMouseDown={handleCanvasMouseDown}
                        onMouseMove={handleCanvasMouseMove}
                        onMouseUp={handleCanvasMouseUp}
                        style={{ maxWidth: '100%', display: 'block', borderRadius: '8px' }}
                      />
                    </div>
                  ) : (
                    <div className="camera-placeholder">
                      <div className="placeholder-icon" style={{ color: '#38bdf8' }}>
                        <Crosshair size={36} />
                      </div>
                      <h4 style={{ color: '#f8fafc', fontWeight: 600 }}>No Image Loaded for Annotation</h4>
                      <p style={{ fontSize: '0.8rem', color: '#94a3b8', maxWidth: '380px', marginTop: '0.35rem' }}>
                        Click <b>Snap Live Camera Frame</b> or <b>Upload Photo</b> to start drawing bounding boxes over your fish or eggs!
                      </p>
                    </div>
                  )}
                </div>

                {/* Annotation Controls Toolbar */}
                {studioImageSrc && (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', background: 'rgba(255, 255, 255, 0.04)', padding: '0.75rem 1rem', borderRadius: '10px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <label style={{ fontSize: '0.825rem', fontWeight: 600, color: '#94a3b8' }}>Target Label:</label>
                      <select
                        value={currentLabel}
                        onChange={(e) => setCurrentLabel(e.target.value)}
                        style={{ padding: '0.4rem 0.75rem', borderRadius: '8px', background: '#0f172a', color: '#ffffff', border: '1px solid var(--border)', fontSize: '0.85rem' }}
                      >
                        {customClasses.map((cls, idx) => (
                          <option key={idx} value={cls}>{cls}</option>
                        ))}
                      </select>
                      <span style={{ fontSize: '0.8rem', color: '#38bdf8', fontWeight: 600 }}>
                        {drawnBoxes.length} box(es) drawn
                      </span>
                    </div>

                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '0.4rem 0.75rem' }} onClick={() => setDrawnBoxes([])}>
                        <RotateCcw size={13} /> Clear Boxes
                      </button>
                      <button className="btn btn-primary" style={{ fontSize: '0.8rem', padding: '0.4rem 0.85rem' }} onClick={handleSaveSample}>
                        <Save size={14} /> Save Sample to AI Dataset
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Control Bar */}
          <div className="viewport-controls">
            <div className="action-btn-group">
              {activeTab === 'camera' && (
                isStreaming ? (
                  <>
                    <button className="btn btn-danger" onClick={stopCamera}>
                      <Pause size={16} /> Stop Camera
                    </button>
                    <button className="btn btn-secondary" onClick={() => setIsPaused(!isPaused)}>
                      {isPaused ? <Play size={16} /> : <Pause size={16} />} {isPaused ? 'Resume' : 'Pause'}
                    </button>
                  </>
                ) : (
                  <button className="btn btn-primary" onClick={startCamera}>
                    <Play size={16} /> Start Live Camera
                  </button>
                )
              )}

              <button className="btn btn-secondary" onClick={handleReset}>
                <RotateCcw size={16} /> Reset Counter
              </button>

              <button className="btn btn-secondary" onClick={exportSessionReport}>
                <Download size={16} /> Export CSV Report
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <span className="mono-font" style={{ fontSize: '0.8rem', color: '#64748b' }}>
                FPS: <span style={{ fontWeight: 700, color: '#0284c7' }}>{fps}</span>
              </span>
            </div>
          </div>
        </section>

        {/* Right Column: Telemetry & AI Fine-Tuning Studio Panel */}
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Main Counter Card */}
          <div className="stat-card" style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', color: 'white' }}>
            <div className="stat-header">
              <span className="stat-title" style={{ color: '#94a3b8' }}>TOTAL UNIQUE FISH COUNT</span>
              <Fish size={20} style={{ color: '#38bdf8' }} />
            </div>
            <div className="stat-value" style={{ color: '#38bdf8', fontSize: '3rem', fontWeight: 800 }}>
              {count}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.75rem', fontSize: '0.8rem', color: '#94a3b8' }}>
              <span>Active in Frame: <strong style={{ color: '#10b981' }}>{activeTracks}</strong></span>
              <span>Mode: <strong style={{ textTransform: 'capitalize' }}>{countingMode.replace('_', ' ')}</strong></span>
            </div>
          </div>

          {activeTab === 'studio' ? (
            /* Custom AI Training Panel */
            <div className="stat-card">
              <div className="stat-header">
                <span className="stat-title">CUSTOM AI TRAINING STUDIO</span>
                <Cpu size={18} style={{ color: '#0284c7' }} />
              </div>

              {studioMessage && (
                <div style={{ marginTop: '0.5rem', padding: '0.5rem 0.75rem', borderRadius: '8px', fontSize: '0.775rem', background: studioMessage.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(2, 132, 199, 0.15)', color: studioMessage.type === 'success' ? '#10b981' : '#0284c7' }}>
                  {studioMessage.text}
                </div>
              )}

              {/* Custom Class Manager */}
              <div style={{ marginTop: '1rem', borderBottom: '1px solid var(--border)', pb: '1rem' }}>
                <label style={{ fontSize: '0.775rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  1. Registered Classes / Species
                </label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.5rem', marginBottom: '0.75rem' }}>
                  {customClasses.map((cls, i) => (
                    <span key={i} className="badge badge-blue" style={{ fontSize: '0.75rem' }}>
                      {cls} ({datasetStats.annotations_per_class[cls] || 0})
                    </span>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  <input
                    type="text"
                    placeholder="New Class (e.g. Fish Egg, Tilapia)..."
                    value={newClassName}
                    onChange={(e) => setNewClassName(e.target.value)}
                    style={{ flex: 1, padding: '0.35rem 0.65rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.8rem' }}
                  />
                  <button className="btn btn-secondary" style={{ padding: '0.35rem 0.65rem', fontSize: '0.775rem' }} onClick={handleAddClass}>
                    <Plus size={14} /> Add
                  </button>
                </div>
              </div>

              {/* Dataset Summary */}
              <div style={{ marginTop: '1rem', borderBottom: '1px solid var(--border)', pb: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label style={{ fontSize: '0.775rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    2. Collected Training Dataset
                  </label>
                  <button style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }} onClick={handleClearDataset}>
                    <Trash2 size={12} /> Clear Dataset
                  </button>
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', marginTop: '0.25rem' }}>
                  {datasetStats.total_samples} Annotated Image(s)
                </div>
              </div>

              {/* AI Fine-Tuning Execution Panel */}
              <div style={{ marginTop: '1rem' }}>
                <label style={{ fontSize: '0.775rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  3. Fine-Tune Custom Model
                </label>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginTop: '0.5rem' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Model Filename:</span>
                    <input
                      type="text"
                      value={trainModelName}
                      onChange={(e) => setTrainModelName(e.target.value)}
                      style={{ width: '100%', padding: '0.35rem 0.65rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.8rem', marginTop: '0.2rem' }}
                    />
                  </div>

                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b' }}>
                      <span>Epochs: {trainEpochs}</span>
                      <span>Fast (10-30 epochs)</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="5"
                      value={trainEpochs}
                      onChange={(e) => setTrainEpochs(e.target.value)}
                      style={{ width: '100%', marginTop: '0.2rem' }}
                    />
                  </div>

                  <button
                    className="btn btn-primary"
                    style={{ width: '100%', justifyContent: 'center', marginTop: '0.4rem', padding: '0.55rem' }}
                    onClick={handleStartTraining}
                    disabled={trainStatus.status === 'training'}
                  >
                    <Sparkles size={16} /> {trainStatus.status === 'training' ? 'Training in Progress...' : '🚀 Start Custom AI Training'}
                  </button>

                  {/* Progress Bar */}
                  {trainStatus.status === 'training' && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 600, color: '#0284c7' }}>
                        <span>{trainStatus.message}</span>
                        <span>{trainStatus.progress_pct}%</span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '999px', overflow: 'hidden', marginTop: '0.25rem' }}>
                        <div style={{ width: `${trainStatus.progress_pct}%`, height: '100%', background: 'linear-gradient(90deg, #0284c7, #06b6d4)', transition: 'width 0.4s ease' }} />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Load Trained Model Switcher */}
              <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                <label style={{ fontSize: '0.775rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  4. Select Live Active AI Model
                </label>
                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem' }}>
                  <select
                    value={selectedModelToLoad}
                    onChange={(e) => setSelectedModelToLoad(e.target.value)}
                    style={{ flex: 1, padding: '0.35rem 0.65rem', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '0.8rem' }}
                  >
                    {availableModels.map((m, i) => (
                      <option key={i} value={m}>{m}</option>
                    ))}
                  </select>
                  <button className="btn btn-secondary" style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }} onClick={() => handleSelectModel(selectedModelToLoad)}>
                    Load Model
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* Settings & Configuration Panel */
            <div className="stat-card">
              <div className="stat-header">
                <span className="stat-title">AI VISION CONFIGURATION</span>
                <Sliders size={18} style={{ color: '#0284c7' }} />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '0.75rem' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.35rem' }}>
                    Counting Strategy
                  </label>
                  <select
                    value={countingMode}
                    onChange={(e) => {
                      setCountingMode(e.target.value);
                      handleSettingsChange(undefined, e.target.value, undefined);
                    }}
                    style={{ width: '100%', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem' }}
                  >
                    <option value="unique_id">Unique ID Tracking (ByteTrack)</option>
                    <option value="line_crossing">Line Crossing Gate (Virtual Line)</option>
                    <option value="roi">Region of Interest (ROI Zone)</option>
                  </select>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
                    <span>Confidence Threshold</span>
                    <span>{Math.round(confThreshold * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.80"
                    step="0.05"
                    value={confThreshold}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setConfThreshold(val);
                      handleSettingsChange(val, undefined, undefined);
                    }}
                    style={{ width: '100%' }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
                    <span>Minimum Verification Hits</span>
                    <span>{minHits} frames</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={minHits}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      setMinHits(val);
                      handleSettingsChange(undefined, undefined, val);
                    }}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
