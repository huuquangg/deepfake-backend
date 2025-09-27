from pydantic import BaseModel

class DetectionRequest(BaseModel):
    video_url: str
    method: str  # "cnn" | "transformer"

class DetectionResponse(BaseModel):
    method: str
    is_fake: bool
    confidence: float
