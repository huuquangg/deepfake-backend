from fastapi import APIRouter, HTTPException
from deepfake_backend.app.features.detection.dtos import DetectionRequest, DetectionResponse
from deepfake_backend.app.features.detection import services
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import os
import json

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

@router.post("/merge-openface-with-mobilenet")
async def merge(
    file: UploadFile = File(...),
    model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"])
):
    try:
        # Read image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Extract embedding from chosen model
        if model_name == "mobilenet":
            embedding = mobilenet_extractor.extract(img)  # 1D list
        else:
            embedding = resnet_extractor.extract(img)     # 1D list

        embedding = np.array(embedding, dtype=np.float32)
        print(f"[DEBUG] {model_name} embedding shape: {embedding.shape}")
        print(f"[DEBUG] {model_name} embedding sample: {embedding[:10]}")  # first 10 values

        # Prepare OpenFace CSV path
        input_file = file.filename
        csv_file_output = os.path.splitext(input_file)[0] + ".csv"
        csv_file_path = f"/home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/extractor/open_face/output/{csv_file_output}"

        # Read OpenFace CSV numeric values (first row only)
        openface_vec = []
        with open(csv_file_path, newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)  # skip header
            first_row = next(reader, None)
            if first_row:
                for value in first_row:
                    try:
                        openface_vec.append(float(value))
                    except ValueError:
                        pass  # ignore non-numeric values

        openface_vec = np.array(openface_vec, dtype=np.float32)
        print(f"[DEBUG] OpenFace vector shape: {openface_vec.shape}")
        print(f"[DEBUG] OpenFace vector sample: {openface_vec[:10]}")  # first 10 values

        embedding = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-8)
        openface_vec = (openface_vec - np.mean(openface_vec)) / (np.std(openface_vec) + 1e-8)

        print(f"[DEBUG] Scaled {model_name} embedding sample: {embedding[:10]}")
        print(f"[DEBUG] Scaled OpenFace vector sample: {openface_vec[:10]}")

        # Merge embeddings
        merged_vec = np.concatenate([embedding, openface_vec], axis=0)
        print(f"[DEBUG] Merged vector shape: {merged_vec.shape}")
        print(f"[DEBUG] Merged vector sample: {merged_vec[:10]}")

        return JSONResponse(content={"merged_vector": merged_vec.tolist(),
                                     "shape": merged_vec.shape})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)