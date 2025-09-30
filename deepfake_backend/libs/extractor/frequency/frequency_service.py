# deepfake_backend/libs/extractor/frequency/frequency_service.py

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import UploadFile
from pydantic import BaseModel
import getpass


# =========================
# DTO trả về cho FastAPI
# =========================
class FrequencyResponse(BaseModel):
    status: bool
    input_file: str
    output_dir: str
    feature_npy: str
    feature_shape: Tuple[int, int, int]  # (C, H, W)
    preview_images: List[str]            # các ảnh PNG xem nhanh


# =========================
# Tiện ích chuẩn hoá
# =========================
def _to_gray_float01(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> gray float32 in [0,1]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    return gray


def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


# =========================
# FFT / DCT
# =========================
def compute_fft_mag(gray01: np.ndarray) -> np.ndarray:
    """
    FFT magnitude (log) chuẩn hoá về [0,1].
    Output shape: (H, W), float32.
    """
    # chuyển về float64 cho FFT ổn định
    x = gray01.astype(np.float64)
    F = np.fft.fft2(x)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    mag_log = np.log1p(mag)  # log(1 + |F|)
    mag01 = _minmax01(mag_log).astype(np.float32)
    return mag01


def compute_dct_mag(gray01: np.ndarray) -> np.ndarray:
    """
    DCT (cv2.dct yêu cầu float32 2D).
    Dùng |coeff| rồi log + chuẩn hoá.
    Output shape: (H, W), float32.
    """
    # đảm bảo contiguous float32
    x = np.ascontiguousarray(gray01.astype(np.float32))
    dct = cv2.dct(x)  # 2D DCT toàn ảnh
    dct_abs = np.abs(dct)
    dct_log = np.log1p(dct_abs)
    dct01 = _minmax01(dct_log).astype(np.float32)
    return dct01


# =========================
# SRM filter bank (30 filters)
# (xấp xỉ bank phổ biến; đủ tốt cho đặc trưng tần số)
# =========================
def _srm_kernels_30() -> List[np.ndarray]:
    k: List[np.ndarray] = []

    # --- 1) Laplacian & LoG variants ---
    k.append(np.array([[0,  -1,  0],
                       [-1,  4, -1],
                       [0,  -1,  0]], np.float32))  # Laplacian 4-neigh
    k.append(np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], np.float32))  # Laplacian 8-neigh

    # 5x5 LoG nhẹ
    k.append((1/256)*np.array([[0,  0, -1,  0,  0],
                               [0, -1, -2, -1,  0],
                               [-1,-2, 16, -2,-1],
                               [0, -1, -2, -1,  0],
                               [0,  0, -1,  0,  0]], np.float32))

    # --- 2) Sobel & Scharr (x/y + biến thể) ---
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], np.float32)
    sobely = sobelx.T
    k += [sobelx, sobely]

    scharrx = np.array([[-3,  0,  3],
                        [-10, 0, 10],
                        [-3,  0,  3]], np.float32)
    scharry = scharrx.T
    k += [scharrx, scharry]

    # --- 3) Prewitt, Roberts (x/y) ---
    prewittx = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], np.float32)
    prewitty = prewittx.T
    k += [prewittx, prewitty]

    robertsx = np.array([[1, 0],
                         [0,-1]], np.float32)
    robertsy = np.array([[0, 1],
                         [-1,0]], np.float32)
    k += [robertsx, robertsy]

    # --- 4) Second-order directional / high-pass nhỏ ---
    k += [
        np.array([[1,-2, 1]], np.float32),              # 1x3
        np.array([[1],[-2],[1]], np.float32),           # 3x1
        np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], np.float32),             # Laplacian alt
        np.array([[2,-1, 0],
                  [-1, 0, 1],
                  [0, 1, 2]], np.float32),             # diagonal-ish
        np.array([[0,-1, 2],
                  [-1, 0, 1],
                  [2, 1, 0]], np.float32),
    ]

    # --- 5) 5x5 edge/HPF variants ---
    k += [
        (1/12)*np.array([[0, 0,-1, 0, 0],
                         [0,-1,-2,-1, 0],
                         [-1,-2,16,-2,-1],
                         [0,-1,-2,-1, 0],
                         [0, 0,-1, 0, 0]], np.float32),

        (1/8)*np.array([[0, 0, 0, 0, 0],
                        [0, 0,-1, 0, 0],
                        [0,-1, 4,-1, 0],
                        [0, 0,-1, 0, 0],
                        [0, 0, 0, 0, 0]], np.float32),

        (1/4)*np.array([[0, 0, 1, 0, 0],
                        [0, 1, 2, 1, 0],
                        [1, 2,-16,2, 1],
                        [0, 1, 2, 1, 0],
                        [0, 0, 1, 0, 0]], np.float32),
    ]

    # --- 6) Một số kernel định hướng 45°/135° ---
    k += [
        np.array([[ 0,  1,  0],
                  [-1,  0,  1],
                  [ 0, -1,  0]], np.float32),

        np.array([[ 0, -1,  0],
                  [ 1,  0, -1],
                  [ 0,  1,  0]], np.float32),

        np.array([[ 1, -2,  1],
                  [-2,  4, -2],
                  [ 1, -2,  1]], np.float32),  # HPF ring

        np.array([[-1,  2, -1],
                  [ 2, -4,  2],
                  [-1,  2, -1]], np.float32),
    ]

    # --- 7) Bổ sung để đủ 30 ---
    k += [
        np.array([[1,-1, 0],
                  [-1,0, 1],
                  [0, 1,-1]], np.float32),

        np.array([[0, 1,-1],
                  [1, 0,-1],
                  [-1,-1, 0]], np.float32),

        np.array([[0, 0, 0],
                  [1,-2, 1],
                  [0, 0, 0]], np.float32),

        np.array([[0, 1, 0],
                  [0,-2, 0],
                  [0, 1, 0]], np.float32),
    ]

    while len(k) < 30:
        k.append(np.array([[0,-1, 0],
                           [-1, 4,-1],
                           [0,-1, 0]], np.float32))

    return k[:30]


def compute_srm_stack(gray01: np.ndarray) -> np.ndarray:
    """
    Chạy 30 kernel SRM-like -> stack (30, H, W), mỗi map chuẩn hoá [0,1].
    """
    kernels = _srm_kernels_30()
    maps: List[np.ndarray] = []
    for ker in kernels:
        # filter2D nhận single-channel ok
        fmap = cv2.filter2D(gray01, ddepth=cv2.CV_32F, kernel=ker)
        fmap = np.abs(fmap)
        fmap = _minmax01(fmap)
        maps.append(fmap.astype(np.float32))
    srm = np.stack(maps, axis=0).astype(np.float32)  # (30, H, W)
    return srm


# =========================
# Service kiểu OpenFaceService
# =========================
class FrequencyService:
    def __init__(
        self,
        base_dir: str | None = None,
        user: str | None = None
    ):
        """
        Service trích xuất đặc trưng miền tần số (FFT, DCT, SRM-30).
        - Portable cho macOS/Linux/Windows.
        - Không dùng sudo/chown mặc định.
        """
        # 1) Home & user hiện tại
        home = Path(os.path.expanduser("~"))            # /Users/tien (macOS) hoặc /home/tien (Linux)
        current_user = getpass.getuser()                # "tien"

        # 2) Cho phép override qua tham số hoặc ENV
        env_base = os.getenv("FREQ_BASE_DIR")
        env_owner = os.getenv("FREQ_OWNER")

        self.base_dir = Path(
            base_dir
            or env_base
            or home / "thesis/deepfake/deepfake_backend/libs/extractor/frequency"
        )
        self.user = user or env_owner or current_user

        # 3) Tạo input/output
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _clear_dir(self, path: str | Path):
        """Dọn sạch thư mục an toàn bằng Python, không shell."""
        p = Path(path)
        if not p.exists():
            return
        for child in p.iterdir():
            if child.is_file() or child.is_symlink():
                try:
                    child.unlink(missing_ok=True)
                except Exception:
                    pass
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    def _save_preview_png(self, arr01: np.ndarray, path_png: str | Path):
        """Lưu ảnh arr01 [0,1] -> PNG."""
        path_png = str(path_png)
        img8 = (np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(path_png, img8)

    def _compute_features(self, img_bgr: np.ndarray, basename: str) -> Tuple[np.ndarray, List[str]]:
        """
        Tính FFT (1), DCT (1), SRM (30) -> stack (32,H,W).
        Lưu một số preview PNG để dễ kiểm tra.
        """
        gray01 = _to_gray_float01(img_bgr)

        # --- FFT & DCT ---
        fft_map = compute_fft_mag(gray01)        # (H, W)
        dct_map = compute_dct_mag(gray01)        # (H, W)

        # --- SRM stack ---
        srm = compute_srm_stack(gray01)          # (30, H, W)

        # --- Stack: (32, H, W) ---
        fft_c = fft_map[None, ...]               # (1, H, W)
        dct_c = dct_map[None, ...]               # (1, H, W)
        feat = np.concatenate([fft_c, dct_c, srm], axis=0).astype(np.float32)

        # Lưu preview
        previews: List[str] = []
        fft_png = self.output_dir / f"{basename}_fft.png"
        dct_png = self.output_dir / f"{basename}_dct.png"
        self._save_preview_png(fft_map, fft_png)
        self._save_preview_png(dct_map, dct_png)
        previews += [str(fft_png), str(dct_png)]

        # thêm 4 kênh SRM đại diện để xem nhanh
        for idx in [0, 1, 2, 3]:
            srm_png = self.output_dir / f"{basename}_srm_{idx:02d}.png"
            self._save_preview_png(srm[idx], srm_png)
            previews.append(str(srm_png))

        return feat, previews

    def analyze_image(self, file: UploadFile) -> FrequencyResponse:
        """
        Nhận 1 ảnh RGB (UploadFile), xoá input/output cũ, tính features,
        lưu .npy và trả thông tin.
        """
        # Clear input/output để không lẫn kết quả cũ
        self._clear_dir(self.input_dir)
        self._clear_dir(self.output_dir)

        # Lưu file input
        input_path = self.input_dir / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Đọc ảnh (BGR, uint8)
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không đọc được ảnh đầu vào (cv2.imread trả None).")

        # Tính đặc trưng
        basename = Path(file.filename).stem
        feat, previews = self._compute_features(img, basename)

        # Lưu .npy (C,H,W) float32
        feat_path = self.output_dir / f"{basename}_freq_features.npy"
        np.save(str(feat_path), feat)

        C, H, W = feat.shape
        return FrequencyResponse(
            status=True,
            input_file=file.filename,
            output_dir=str(self.output_dir),
            feature_npy=str(feat_path),
            feature_shape=(C, H, W),
            preview_images=previews
        )
