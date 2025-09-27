from fastapi import APIRouter, HTTPException
from deepfake_backend.app.features.detection.dtos import DetectionRequest, DetectionResponse
from deepfake_backend.app.features.detection import services
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io

from deepfake_backend.libs.extractor.mobilenet.mobilenet import MobileNetExtractor
from deepfake_backend.libs.extractor.resnet.resnet import ResNetExtractor
from deepfake_backend.libs.extractor.open_face.open_face import OpenFaceService
# Initialize models once
mobilenet_extractor = MobileNetExtractor()
resnet_extractor = ResNetExtractor()
openface_service = OpenFaceService(container_name="openface")

router = APIRouter()

@router.post("/", response_model=DetectionResponse)
def detect_video(request: DetectionRequest):
    try:
        return services.detect_deepfake(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"])
):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if model_name == "mobilenet":
            embedding = mobilenet_extractor.extract(img)
        else:
            embedding = resnet_extractor.extract(img)

        return JSONResponse(content={"embedding": embedding, "model": model_name})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    return openface_service.analyze_image(file)
