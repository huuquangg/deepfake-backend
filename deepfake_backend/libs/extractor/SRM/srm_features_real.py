import os
import cv2
import csv
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

# =========================
# CẤU HÌNH CƠ BẢN
# =========================
INPUT_DIR = r"/Applications/Tien/deepfake/Dataset/celeb_df_crop/real"
OUTPUT_DIR = r"/Applications/Tien/deepfake/extract-celeb/SRM"
OUTPUT_CSV = "srm_features_real.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_SIZE = (256, 256)     # resize về 256x256 cho ổn định
CLAMP_T = 3.0                # clamp residual vào [-3, 3]
ENTROPY_BINS = 41            # số bins khi tính entropy trong [-3, 3]

# =========================
# KERNELS SRM / HIGH-PASS
# (Bạn có thể mở rộng thêm)
# =========================
def get_srm_kernels() -> List[np.ndarray]:
    k = []

    # Laplacian (4-neighbors)
    k.append(np.array([[0, -1,  0],
                       [-1, 4, -1],
                       [0, -1,  0]], dtype=np.float32))

    # Laplacian (8-neighbors)
    k.append(np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32))

    # High-pass (1, -2, 1) 2D
    k.append(np.array([[ 1, -2,  1],
                       [-2,  4, -2],
                       [ 1, -2,  1]], dtype=np.float32))

    # 2nd derivative horizontal & vertical
    k.append(np.array([[ 0,  0,  0],
                       [ 1, -2,  1],
                       [ 0,  0,  0]], dtype=np.float32))  # H second-deriv
    k.append(np.array([[ 0,  1,  0],
                       [ 0, -2,  0],
                       [ 0,  1,  0]], dtype=np.float32))  # V second-deriv

    # Diagonal second-derivative (2 hướng)
    k.append(np.array([[ 1,  0,  0],
                       [ 0, -2,  0],
                       [ 0,  0,  1]], dtype=np.float32))
    k.append(np.array([[ 0,  0,  1],
                       [ 0, -2,  0],
                       [ 1,  0,  0]], dtype=np.float32))

    # Sobel X / Y
    k.append(np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32))
    k.append(np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32))

    # Scharr X / Y (nhạy cạnh hơn Sobel)
    k.append(np.array([[-3, 0, 3],
                       [-10, 0, 10],
                       [-3, 0, 3]], dtype=np.float32))
    k.append(np.array([[-3, -10, -3],
                       [ 0,   0,  0],
                       [ 3,  10,  3]], dtype=np.float32))

    # Predictive residual (trừ lân cận gần) – 4 hướng
    k.append(np.array([[0, 0, 0],
                       [1,-1, 0],
                       [0, 0, 0]], dtype=np.float32))   # from left
    k.append(np.array([[0, 0, 0],
                       [0,-1, 1],
                       [0, 0, 0]], dtype=np.float32))   # from right
    k.append(np.array([[0, 1, 0],
                       [0,-1, 0],
                       [0, 0, 0]], dtype=np.float32))   # from up
    k.append(np.array([[0, 0, 0],
                       [0,-1, 0],
                       [0, 1, 0]], dtype=np.float32))   # from down

    # Predictive residual – 4 đường chéo
    k.append(np.array([[1, 0, 0],
                       [0,-1, 0],
                       [0, 0, 0]], dtype=np.float32))   # from up-left
    k.append(np.array([[0, 0, 1],
                       [0,-1, 0],
                       [0, 0, 0]], dtype=np.float32))   # from up-right
    k.append(np.array([[0, 0, 0],
                       [0,-1, 0],
                       [1, 0, 0]], dtype=np.float32))   # from down-left
    k.append(np.array([[0, 0, 0],
                       [0,-1, 0],
                       [0, 0, 1]], dtype=np.float32))   # from down-right

    # 5x5 Laplacian of Gaussian (LoG) – nhấn mạnh biên / texture
    k.append(np.array([[ 0,  0, -1,  0,  0],
                       [ 0, -1, -2, -1,  0],
                       [-1, -2, 16, -2, -1],
                       [ 0, -1, -2, -1,  0],
                       [ 0,  0, -1,  0,  0]], dtype=np.float32))

    # Một số high-pass 5x5 đơn giản
    k.append(np.array([[ 0,  0, -1,  0,  0],
                       [ 0, -1, -2, -1,  0],
                       [-1, -2, 12, -2, -1],
                       [ 0, -1, -2, -1,  0],
                       [ 0,  0, -1,  0,  0]], dtype=np.float32))

    # Bạn có thể thêm nhiều kernel SRM chuẩn hơn tại đây...
    return k

# =========================
# HÀM TÍNH TOÁN ĐẶC TRƯNG
# =========================
def clamp_residual(residual: np.ndarray, t: float = 3.0) -> np.ndarray:
    return np.clip(residual, -t, t)

def moments_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    # mean, variance, skewness, kurtosis (Fisher)
    x = x.astype(np.float64)
    mu = x.mean()
    var = x.var()
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return float(mu), float(var), 0.0, -3.0  # skew=0, kurtosis=-3 (theo Fisher) khi std=0

    x_centered = x - mu
    m3 = np.mean(x_centered**3)
    m4 = np.mean(x_centered**4)
    skew = m3 / (std**3)
    kurt = m4 / (std**4) - 3.0
    return float(mu), float(var), float(skew), float(kurt)

def shannon_entropy(x: np.ndarray, bins: int = 41, clamp: float = 3.0) -> float:
    # Tính entropy trên histogram trong [-clamp, clamp]
    hist, _ = np.histogram(x, bins=bins, range=(-clamp, clamp), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def energy(x: np.ndarray) -> float:
    # Mean of squares (ổn định theo kích thước ảnh)
    return float(np.mean(x.astype(np.float64)**2))

def extract_features_for_image(img_gray: np.ndarray, kernels: List[np.ndarray]) -> List[float]:
    feats = []
    for ker in kernels:
        residual = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=ker, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_REFLECT)
        residual = clamp_residual(residual, CLAMP_T)

        mu, var, skew, kurt = moments_stats(residual)
        ent = shannon_entropy(residual, bins=ENTROPY_BINS, clamp=CLAMP_T)
        eng = energy(residual)

        feats.extend([mu, var, skew, kurt, ent, eng])
    return feats

# =========================
# TIỆN ÍCH
# =========================
def list_images(root: str) -> List[Path]:
    paths = []
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"INPUT_DIR không tồn tại: {root}")
    for p in sorted(root_p.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths

def build_header(num_kernels: int) -> List[str]:
    header = ["filename"]
    for i in range(1, num_kernels + 1):
        header += [
            f"SRM_mean_{i}", f"SRM_var_{i}", f"SRM_skew_{i}",
            f"SRM_kurt_{i}", f"SRM_entropy_{i}", f"SRM_energy_{i}"
        ]
    return header

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

    kernels = get_srm_kernels()
    num_k = len(kernels)
    header = build_header(num_k)

    img_paths = list_images(INPUT_DIR)
    if len(img_paths) == 0:
        print("⚠️  Không tìm thấy ảnh nào trong INPUT_DIR.")
        return

    print(f"📂 Tìm thấy {len(img_paths)} ảnh. Bắt đầu trích xuất SRM với {num_k} kernels ...")
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, p in enumerate(img_paths, 1):
            try:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  • Bỏ qua (không đọc được): {p}")
                    continue

                # Grayscale + resize
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if TARGET_SIZE is not None:
                    gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                feats = extract_features_for_image(gray, kernels)
                row = [p.name] + feats
                writer.writerow(row)

                if idx % 100 == 0:
                    print(f"  ✓ Đã xử lý {idx}/{len(img_paths)} ảnh")

            except Exception as e:
                print(f"  ✗ Lỗi với {p}: {e}")

    print(f"✅ Hoàn tất. CSV lưu tại: {out_csv_path}")
    print(f"👉 Header gồm 1 + {num_k}×6 = {1 + num_k*6} cột.")

if __name__ == "__main__":
    main()
