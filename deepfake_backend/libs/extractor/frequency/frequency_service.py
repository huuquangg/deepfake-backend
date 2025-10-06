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
    feature_txt: str  # ← THÊM: path đến file .txt
    feature_shape: Tuple[int, int, int]  # (H, W, C)
    preview_images: List[str]


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
    x = gray01.astype(np.float64)
    F = np.fft.fft2(x)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    mag_log = np.log1p(mag)
    mag01 = _minmax01(mag_log).astype(np.float32)
    return mag01


def compute_dct_mag(gray01: np.ndarray) -> np.ndarray:
    """
    DCT (cv2.dct yêu cầu float32 2D).
    Dùng |coeff| rồi log + chuẩn hoá.
    Output shape: (H, W), float32.
    """
    x = np.ascontiguousarray(gray01.astype(np.float32))
    dct = cv2.dct(x)
    dct_abs = np.abs(dct)
    dct_log = np.log1p(dct_abs)
    dct01 = _minmax01(dct_log).astype(np.float32)
    return dct01


# =========================
# SRM filter bank (30 filters)
# =========================
def _srm_kernels_30() -> List[np.ndarray]:
    k: List[np.ndarray] = []

    k.append(np.array([[0,  -1,  0],
                       [-1,  4, -1],
                       [0,  -1,  0]], np.float32))
    k.append(np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], np.float32))

    k.append((1/256)*np.array([[0,  0, -1,  0,  0],
                               [0, -1, -2, -1,  0],
                               [-1,-2, 16, -2,-1],
                               [0, -1, -2, -1,  0],
                               [0,  0, -1,  0,  0]], np.float32))

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

    k += [
        np.array([[1,-2, 1]], np.float32),
        np.array([[1],[-2],[1]], np.float32),
        np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], np.float32),
        np.array([[2,-1, 0],
                  [-1, 0, 1],
                  [0, 1, 2]], np.float32),
        np.array([[0,-1, 2],
                  [-1, 0, 1],
                  [2, 1, 0]], np.float32),
    ]

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

    k += [
        np.array([[ 0,  1,  0],
                  [-1,  0,  1],
                  [ 0, -1,  0]], np.float32),

        np.array([[ 0, -1,  0],
                  [ 1,  0, -1],
                  [ 0,  1,  0]], np.float32),

        np.array([[ 1, -2,  1],
                  [-2,  4, -2],
                  [ 1, -2,  1]], np.float32),

        np.array([[-1,  2, -1],
                  [ 2, -4,  2],
                  [-1,  2, -1]], np.float32),
    ]

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
        fmap = cv2.filter2D(gray01, ddepth=cv2.CV_32F, kernel=ker)
        fmap = np.abs(fmap)
        fmap = _minmax01(fmap)
        maps.append(fmap.astype(np.float32))
    srm = np.stack(maps, axis=0).astype(np.float32)
    return srm


# =========================
# Service
# =========================
class FrequencyService:
    def __init__(
        self,
        base_dir: str | None = None,
        user: str | None = None
    ):
        """
        Service trích xuất đặc trưng miền tần số (FFT, DCT, SRM-30).
        """
        home = Path(os.path.expanduser("~"))
        current_user = getpass.getuser()

        env_base = os.getenv("FREQ_BASE_DIR")
        env_owner = os.getenv("FREQ_OWNER")

        self.base_dir = Path(
            base_dir
            or env_base
            or "/Applications/Tien/deepfake-backend/deepfake-backend/deepfake_backend/libs/extractor/frequency"
        )
        self.user = user or env_owner or current_user

        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _clear_dir(self, path: str | Path):
        """Dọn sạch thư mục."""
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

    def _save_features_txt(self, feat: np.ndarray, basename: str) -> str:
        """
        Lưu features dạng text để xem được bằng notepad/text editor.
        Lưu 3 channels đầu (FFT, DCT, SRM_00) với full 224x224 ma trận.
        """
        txt_path = self.output_dir / f"{basename}_freq_features.txt"
        
        H, W, C = feat.shape
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("FREQUENCY DOMAIN FEATURES - NUMERICAL DATA\n")
            f.write("=" * 80 + "\n")
            f.write(f"Shape: (Height={H}, Width={W}, Channels={C})\n")
            f.write(f"Total features: {H * W * C:,}\n")
            f.write(f"Value range: [0.0, 1.0] (normalized)\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Min:    {feat.min():.8f}\n")
            f.write(f"  Max:    {feat.max():.8f}\n")
            f.write(f"  Mean:   {feat.mean():.8f}\n")
            f.write(f"  Std:    {feat.std():.8f}\n")
            f.write(f"  Median: {np.median(feat):.8f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Channel names
            channel_names = ["FFT", "DCT", "SRM_00"]
            
            # Lưu 3 channels đầu (FFT, DCT, SRM_00)
            # Nếu muốn lưu tất cả 32 channels, đổi range(3) thành range(C)
            for ch_idx in range(min(3, C)):
                f.write(f"CHANNEL {ch_idx}: {channel_names[ch_idx]}\n")
                f.write("-" * 80 + "\n")
                
                channel_data = feat[:, :, ch_idx]
                
                # Channel statistics
                f.write(f"Min: {channel_data.min():.8f}  ")
                f.write(f"Max: {channel_data.max():.8f}  ")
                f.write(f"Mean: {channel_data.mean():.8f}\n\n")
                
                # Full 224x224 matrix
                f.write(f"Matrix {H}x{W}:\n")
                for row_idx, row in enumerate(channel_data):
                    # Format: mỗi số 10 ký tự, 6 chữ số thập phân
                    row_str = " ".join(f"{val:10.6f}" for val in row)
                    f.write(f"{row_str}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Footer note
            f.write("NOTE:\n")
            f.write(f"- Only showing first 3 channels (FFT, DCT, SRM_00) out of {C} total channels\n")
            f.write(f"- Full data available in .npy file: {basename}_freq_features.npy\n")
            f.write(f"- File size: ~{os.path.getsize(txt_path) / 1024 / 1024:.2f} MB\n")
        
        return str(txt_path)

    def _compute_features(self, img_bgr: np.ndarray, basename: str) -> Tuple[np.ndarray, List[str]]:
        """
        Tính FFT (1), DCT (1), SRM (30) -> stack (H, W, 32).
        """
        gray01 = _to_gray_float01(img_bgr)

        fft_map = compute_fft_mag(gray01)
        dct_map = compute_dct_mag(gray01)
        srm = compute_srm_stack(gray01)

        fft_c = fft_map[None, ...]
        dct_c = dct_map[None, ...]
        feat = np.concatenate([fft_c, dct_c, srm], axis=0).astype(np.float32)
        
        feat = np.transpose(feat, (1, 2, 0))

        previews: List[str] = []
        fft_png = self.output_dir / f"{basename}_fft.png"
        dct_png = self.output_dir / f"{basename}_dct.png"
        self._save_preview_png(fft_map, fft_png)
        self._save_preview_png(dct_map, dct_png)
        previews += [str(fft_png), str(dct_png)]

        for idx in [0, 1, 2, 3]:
            srm_png = self.output_dir / f"{basename}_srm_{idx:02d}.png"
            self._save_preview_png(srm[idx], srm_png)
            previews.append(str(srm_png))

        return feat, previews

    def analyze_image(self, file: UploadFile) -> FrequencyResponse:
        """
        Nhận 1 ảnh RGB (UploadFile), xoá input/output cũ, tính features,
        lưu .npy, .txt và trả thông tin.
        """
        self._clear_dir(self.input_dir)
        self._clear_dir(self.output_dir)

        input_path = self.input_dir / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không đọc được ảnh đầu vào (cv2.imread trả None).")
        
        if img.shape[0] < 32 or img.shape[1] < 32:
            raise ValueError(f"Ảnh quá nhỏ: {img.shape}. Cần ít nhất 32x32 pixels.")
        
        if img.shape[0] > 4096 or img.shape[1] > 4096:
            raise ValueError(f"Ảnh quá lớn: {img.shape}. Giới hạn 4096x4096 pixels.")
        
        img = cv2.resize(img, (224, 224))

        basename = Path(file.filename).stem
        feat, previews = self._compute_features(img, basename)

        # Lưu .npy
        feat_path = self.output_dir / f"{basename}_freq_features.npy"
        np.save(str(feat_path), feat)
        
        # Lưu .txt
        txt_path = self._save_features_txt(feat, basename)

        H, W, C = feat.shape
        return FrequencyResponse(
            status=True,
            input_file=file.filename,
            output_dir=str(self.output_dir),
            feature_npy=str(feat_path),
            feature_txt=txt_path,
            feature_shape=(H, W, C),
            preview_images=previews
        )