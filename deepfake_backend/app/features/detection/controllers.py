from __future__ import annotations

import io, os, re, glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from fastapi import APIRouter, HTTPException, File, UploadFile, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from deepfake_backend.app.features.detection.dtos import DetectionRequest, DetectionResponse
from deepfake_backend.app.features.detection import services

from deepfake_backend.libs.extractor.mobilenet.mobilenet import MobileNetExtractor
from deepfake_backend.libs.extractor.resnet.resnet import ResNetExtractor
from deepfake_backend.libs.extractor.open_face.open_face import OpenFaceService

# ---------- Config ----------
OPENFACE_OUTPUT_DIR = "/home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/extractor/open_face/output"
MAX_SIDE = 4096  # chặn ảnh quá lớn
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- Init ----------
mobilenet_extractor = MobileNetExtractor()
resnet_extractor = ResNetExtractor()
openface_service = OpenFaceService(container_name="openface")

router = APIRouter()


# ---------- Detection ----------
@router.post("/", response_model=DetectionResponse)
def detect_video(request: DetectionRequest):
    try:
        return services.detect_deepfake(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Extract raw CNN embedding ----------
@router.post("/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"])
):
    try:
        img = await _load_image(file)
        if model_name == "mobilenet":
            emb = await run_in_threadpool(mobilenet_extractor.extract, img)
        else:
            emb = await run_in_threadpool(resnet_extractor.extract, img)
        return JSONResponse(content={"embedding": emb, "model": model_name})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------- Trigger OpenFace (nếu bạn cần) ----------
@router.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    # Gọi sync service phía OpenFace (viết/ghi CSV ngoài ổ đĩa)
    return openface_service.analyze_image(file)


# ---------- OpenFace schema ----------
def of_fixed_columns() -> List[str]:
    cols: List[str] = []
    cols += ["frame", "face_id", "timestamp", "confidence", "success"]
    cols += ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_1_x","gaze_1_y","gaze_1_z","gaze_angle_x","gaze_angle_y"]
    cols += [f"eye_lmk_x_{i}" for i in range(56)]
    cols += [f"eye_lmk_y_{i}" for i in range(56)]
    cols += [f"eye_lmk_X_{i}" for i in range(56)]
    cols += [f"eye_lmk_Y_{i}" for i in range(56)]
    cols += [f"eye_lmk_Z_{i}" for i in range(56)]
    cols += ["pose_Tx","pose_Ty","pose_Tz","pose_Rx","pose_Ry","pose_Rz"]
    cols += [f"x_{i}" for i in range(68)]
    cols += [f"y_{i}" for i in range(68)]
    cols += [f"X_{i}" for i in range(68)]
    cols += [f"Y_{i}" for i in range(68)]
    cols += [f"Z_{i}" for i in range(68)]
    au_r = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU45"]
    au_c = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU28","AU45"]
    cols += [f"{a}_r" for a in au_r]
    cols += [f"{a}_c" for a in au_c]
    return cols

OF_COLS = of_fixed_columns()
OF_DIM = len(OF_COLS)


# ---------- Helpers ----------
async def _load_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        ratio = MAX_SIDE / float(max(w, h))
        img = img.resize((int(w * ratio), int(h * ratio)))
    return img


_norm_re = re.compile(r'[^a-z0-9]+')
def _norm_key(p: str) -> str:
    base = os.path.splitext(os.path.basename(p))[0].lower()
    return _norm_re.sub('', base)

def find_openface_csv(image_filename: str, base_dir: str = OPENFACE_OUTPUT_DIR) -> Optional[str]:
    """
    1) Thử trùng tên trực tiếp: {output_root}/{stem}.csv
    2) Nếu không có, fuzzy trong toàn bộ cây (dừng ở match đầu tiên).
    """
    stem = os.path.splitext(os.path.basename(image_filename))[0]
    direct = os.path.join(base_dir, stem + ".csv")
    if os.path.isfile(direct):
        return direct

    key_img = _norm_key(stem)
    for p in glob.iglob(os.path.join(base_dir, "**", "*.csv"), recursive=True):
        k = _norm_key(p)
        if key_img == k or key_img in k or k in key_img:
            return p
    return None


def zscore_clip(v: np.ndarray, clip: float = 5.0) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    m = np.nanmean(v)
    s = np.nanstd(v)
    v = (v - m) / (s + 1e-8)
    v = np.nan_to_num(v, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None:
        v = np.clip(v, -clip, clip)
    return v


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / (n + 1e-6)).astype(np.float32)


def read_openface_vector(csv_path: Optional[str], require_success: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Đọc một vector OpenFace theo schema cố định:
    - Ưu tiên hàng success==1 & confidence cao nhất (nếu có cột).
    - Trả meta để debug.
    """
    meta: Dict[str, Any] = {"of_ok": 0, "confidence": None, "picked_frame": None, "csv_path": csv_path}
    if not csv_path or (not os.path.isfile(csv_path)):
        meta["error"] = "csv_not_found"
        return np.full((OF_DIM,), np.nan, dtype=np.float32), meta

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        meta["error"] = f"read_csv: {e}"
        return np.full((OF_DIM,), np.nan, dtype=np.float32), meta

    if df.empty:
        meta["error"] = "empty_csv"
        return np.full((OF_DIM,), np.nan, dtype=np.float32), meta

    # Lọc & chọn hàng tốt
    cols = df.columns.tolist()
    pick = df.copy()
    if require_success and "success" in cols:
        pick = pick[pick["success"] == 1] if not pick.empty else pick
    if "confidence" in cols:
        pick = pick.sort_values(["confidence"], ascending=[False])

    # Nếu sau lọc rỗng, fallback toàn bảng
    if pick.empty:
        pick = df

    row = pick.iloc[0].to_dict()
    vec = np.empty((OF_DIM,), dtype=np.float32)
    for i, k in enumerate(OF_COLS):
        try:
            vec[i] = float(row.get(k, np.nan))
        except Exception:
            vec[i] = np.nan

    meta["of_ok"] = 1 if float(row.get("success", 0)) == 1 else 0
    meta["confidence"] = float(row.get("confidence", np.nan)) if "confidence" in cols else None
    meta["picked_frame"] = int(row.get("frame", -1)) if "frame" in cols else None
    return vec, meta


# ---------- Merge endpoint (chuẩn hoá & ít nhiễu) ----------
@router.post("/merge-openface-with-mobilenet")
async def merge(
    file: UploadFile = File(...),
    model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"]),
    include_vector: bool = Query(True, description="Trả merged_vector hay không (mặc định: True)"),
    min_confidence: float = Query(0.0, description="Ngưỡng confidence tối thiểu của OpenFace; 0.0 = không áp"),
    require_success: bool = Query(True, description="Chỉ nhận frame success==1"),
):
    try:
        # 1) Load ảnh (bảo vệ kích thước)
        img = await _load_image(file)

        # 2) CNN embedding (chạy trong threadpool để không block event loop)
        if model_name == "mobilenet":
            cnn_list = await run_in_threadpool(mobilenet_extractor.extract, img)   # ~1280-d list
        else:
            cnn_list = await run_in_threadpool(resnet_extractor.extract, img)      # ~2048-d list
        cnn = np.asarray(cnn_list, dtype=np.float32)

        # 3) Tìm & đọc OpenFace CSV
        csv_path = find_openface_csv(file.filename, base_dir=OPENFACE_OUTPUT_DIR)
        of_raw, of_meta = read_openface_vector(csv_path, require_success=require_success)

        # 3.1) Gate theo confidence nếu cần
        of_conf = float(of_meta.get("confidence") or 0.0)
        csv_exists = bool(csv_path and os.path.isfile(csv_path))
        if not csv_exists:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "openface_csv_not_found", "filename": file.filename}
            )
        if min_confidence > 0.0 and (np.isnan(of_conf) or of_conf < min_confidence):
            return JSONResponse(
                status_code=422,
                content={"ok": False, "error": "low_confidence", "confidence": of_conf, "min_confidence": min_confidence}
            )

        # 4) Chuẩn hoá để giảm nhiễu
        #    - OpenFace: zscore + clip + NaN→0
        #    - CNN: zscore nhẹ rồi L2 để cân scale giữa các ảnh/thiết bị
        of_vec = zscore_clip(of_raw, clip=5.0)
        cnn = l2norm(zscore_clip(cnn, clip=5.0))

        # 4.1) Quality checks (tránh vector hằng số)
        if not np.isfinite(of_vec).any():
            return JSONResponse(status_code=422, content={"ok": False, "error": "openface_vector_nan_only"})
        if float(np.std(cnn)) < 1e-6:
            return JSONResponse(status_code=422, content={"ok": False, "error": "cnn_vector_constant_like"})

        # 5) Merge theo thứ tự cố định [OpenFace | CNN]
        merged_vec = np.concatenate([of_vec, cnn], axis=0).astype(np.float32)

        payload = {
            "ok": True,
            "filename": file.filename,
            "schema_version": "OFv1_CNNv1",
            "dims": {
                "openface": int(of_vec.shape[0]),
                "cnn": int(cnn.shape[0]),
                "total": int(merged_vec.shape[0]),
            },
            "openface": {
                "csv_found": csv_exists,
                "csv_path": csv_path,
                "of_ok": int(of_meta.get("of_ok") or 0),
                "confidence": None if np.isnan(of_conf) else float(of_conf),
                "picked_frame": int(of_meta.get("picked_frame") or -1),
            },
            "model": model_name,
        }

        if include_vector:
            payload["merged_vector"] = merged_vec.tolist()

        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)
