from fastapi import FastAPI
from deepfake_backend.app.features.detection.controllers import router as detection_router

def include_routers(app: FastAPI):
    app.include_router(detection_router, prefix="/api/detection", tags=["detection"])
