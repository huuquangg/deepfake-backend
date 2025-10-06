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
import os  # ← CẦN CÓ DÒNG NÀY
from pathlib import Path  # ← VÀ DÒNG NÀY

from deepfake_backend.libs.extractor.mobilenet.mobilenet import MobileNetExtractor
from deepfake_backend.libs.extractor.resnet.resnet import ResNetExtractor
from deepfake_backend.libs.extractor.frequency.frequency_service import (
    FrequencyService, FrequencyResponse
)
# from deepfake_backend.libs.extractor.open_face.open_face import OpenFaceService
# Initialize models once
mobilenet_extractor = MobileNetExtractor()
resnet_extractor = ResNetExtractor()
# openface_service = OpenFaceService(container_name="openface")

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


# ====== KHỞI TẠO SERVICE 1 LẦN ======
freq_service = FrequencyService()

# ====== ENDPOINT TEST FREQUENCY SERVICE ======
@router.post("/frequency/analyze", response_model=FrequencyResponse)
async def analyze_frequency(file: UploadFile = File(...)):
    try:
        return freq_service.analyze_image(file)
    except ValueError as e:
        # lỗi input ảnh
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # lỗi tính toán/IO khác
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/merge-frequency-with-mobilenet")
async def merge_frequency_mobilenet(
    file: UploadFile = File(...),
    model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"])
):
    try:
        # Read image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Extract embedding
        if model_name == "mobilenet":
            embedding = mobilenet_extractor.extract(img)
        else:
            embedding = resnet_extractor.extract(img)
        
        embedding = np.array(embedding, dtype=np.float32)
        print(f"[DEBUG] {model_name} embedding shape: {embedding.shape}")
        
        # Path cụ thể cho máy bạn
        base_dir = Path("/Applications/Tien/deepfake-backend/deepfake-backend/deepfake_backend/libs/extractor/frequency")
        output_dir = base_dir / "output"
        
        # Prepare Frequency features path
        input_file = file.filename
        npy_file_output = os.path.splitext(input_file)[0] + "_freq_features.npy"
        npy_file_path = output_dir / npy_file_output
        
        print(f"[DEBUG] Looking for file: {npy_file_path}")
        
        # Read Frequency features
        freq_features = np.load(str(npy_file_path))  # (32, H, W)
        print(f"[DEBUG] Frequency features shape: {freq_features.shape}")
        
        # Global Average Pooling
        freq_vec = np.mean(freq_features, axis=(1, 2)).astype(np.float32)
        print(f"[DEBUG] Frequency vector shape: {freq_vec.shape}")
        
        # Normalize
        embedding = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-8)
        freq_vec = (freq_vec - np.mean(freq_vec)) / (np.std(freq_vec) + 1e-8)
        
        # Merge
        merged_vec = np.concatenate([freq_vec, embedding], axis=0)
        print(f"[DEBUG] Merged vector shape: {merged_vec.shape}")
        
        return JSONResponse(content={
            "status": True,
            "merged_vector": merged_vec.tolist(),
            "shape": list(merged_vec.shape),
            "frequency_dim": int(freq_vec.shape[0]),
            "embedding_dim": int(embedding.shape[0]),
            "total_dim": int(merged_vec.shape[0])
        })
        
    except FileNotFoundError as e:
        return JSONResponse(
            content={"error": f"Frequency features file not found: {npy_file_path}"}, 
            status_code=404
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

# @router.post("/analyze-frame")
# async def analyze_frame(file: UploadFile = File(...)):
#     return openface_service.analyze_image(file)

# @router.post("/merge-openface-with-mobilenet")
# async def merge(
#     file: UploadFile = File(...),
#     model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"])
# ):
#     try:
#         img_bytes = await file.read()
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

#         if model_name == "mobilenet":
#             embedding = mobilenet_extractor.extract(img)
#         else:
#             embedding = resnet_extractor.extract(img)

#         # print(f"mobilenet_vec: {embedding}")


#         csv_file = "/home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/extractor/open_face/output/sample1.csv"

#         openface_vec = []

#         with open(csv_file, newline='') as f:
#             reader = csv.reader(f)
#             next(reader)  # skip header
#             for row in reader:
#                 for value in row:
#                     try:
#                         num = float(value)
#                         openface_vec.append(num)
#                     except ValueError:
#                         # ignore empty or non-numeric values
#                         pass


#         scaler_mn = StandardScaler()
#         scaler_of = StandardScaler()
        
#         print(f"embedding: {embedding}")
#         print(f"openface_vec: {openface_vec}")
#         embedding = np.array(embedding)
#         openface_vec = np.array(openface_vec)
#         print(f"embedding: {embedding}")
#         print(f"openface_vec: {openface_vec}")
        
#         mobilenet_scaled = scaler_mn.fit_transform(embedding.reshape(1, -1))
#         openface_scaled = scaler_of.fit_transform(openface_vec.reshape(1, -1))

#         merged_scaled = np.concatenate([mobilenet_scaled, openface_scaled], axis=1)
#         print(merged_scaled)  # (1, 1958)

#         return JSONResponse(content={"merged_vec": merged_scaled.shape, "model": model_name})

    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)