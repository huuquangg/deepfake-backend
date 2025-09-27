from fastapi import APIRouter, HTTPException
from deepfake_backend.app.features.detection.dtos import DetectionRequest, DetectionResponse
from deepfake_backend.app.features.detection import services

router = APIRouter()

@router.post("/", response_model=DetectionResponse)
def detect_video(request: DetectionRequest):
    try:
        return services.detect_deepfake(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
