from deepfake_backend.app.features.detection.repositories import run_detection
from deepfake_backend.app.features.detection.dtos import DetectionRequest, DetectionResponse

def detect_deepfake(request: DetectionRequest) -> DetectionResponse:
    result = run_detection(request.video_url, request.method)
    return DetectionResponse(
        method=request.method,
        is_fake=result["is_fake"],
        confidence=result["confidence"]
    )
