"""
SatoLeo Count - FastAPI Application Entrypoint
Aquaculture fish and fingerling counting service.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.api.routes_count import router as count_router
from backend.api.routes_stream import router as stream_router
from backend.api.routes_train import router as train_router
from backend.api.models import HealthResponse

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("satoleo.main")

app = FastAPI(
    title="SatoLeo Count API",
    description="Aquaculture fish and fingerling detection, ByteTrack tracking, and unique counting engine.",
    version="1.0.0"
)

# Enable CORS for React frontend (development and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure video outputs directory exists and mount static route
outputs_dir = os.path.abspath("videos/outputs")
os.makedirs(outputs_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# Include Routers
app.include_router(count_router)
app.include_router(stream_router)
app.include_router(train_router)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health status check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="SatoLeo Count AI Engine"
    )

@app.get("/", tags=["Root"])
async def root_index():
    return {
        "message": "Welcome to SatoLeo Count API",
        "docs": "/docs",
        "health": "/health",
        "status_endpoint": "/api/count/status"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting SatoLeo Count FastAPI Server on http://0.0.0.0:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
