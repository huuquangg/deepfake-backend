# -*- coding: utf-8 -*-
"""
Trích xuất đặc trưng DCT (bản Plus, không label)
- Ảnh -> Y (YCbCr) -> resize 256x256 -> level-shift (-128)
- Chia block 8x8 -> DCT-II -> magnitude A, log1p(A) = X
- Chia 3 dải: Low(3x3), Mid(vòng 6x6 trừ 3x3), High(phần còn lại)
- Tính:
  + 12 thống kê: mean/var/skew/kurt (trên X) cho Low/Mid/High
  + 3 entropy (Shannon, trên X) cho Low/Mid/High
  + 1 energy_total = sum(A^2) (toàn ảnh)
  + 20 zig-zag đầu (trên A), mean-pooling theo block
  + Histogram 8 bins (trên X) cho Low/Mid/High (dùng bin edges cố định)
- Ghi CSV: không có label, có filename
- Lưu bin_edges ra JSON để tái lập

Yêu cầu:
  pip install opencv-python numpy scipy pandas tqdm
"""

import os
import json
import math
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Thử dùng scipy để tính skew/kurt; nếu không có thì dùng fallback
try:
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ========= Cấu hình đường dẫn =========
INPUT_DIR = r"/Applications/Tien/deepfake/Dataset/celeb_df_crop/fake"
OUTPUT_DIR = r"/Applications/Tien/deepfake/extract-celeb/DCT"
CSV_PATH  = os.path.join(OUTPUT_DIR, "dct_features_fake.csv")
BIN_JSON_PATH = os.path.join(OUTPUT_DIR, "dct_hist_bin_edges.json")

# ========= Cấu hình thuật toán =========
TARGET_SIZE = (256, 256)      # resize về 256x256
BLOCK = 8                      # kích thước block DCT
LOW_SIZE = 3                   # Low: 3x3 góc trái
MID_SIZE = 6                   # Mid: khung 6x6 trừ 3x3
HIST_BINS = 8                  # số bins cho histogram theo dải
ENTROPY_BINS = 16              # bins nội bộ cho entropy theo dải
ZIGZAG_TAKE = 20               # lấy 20 hệ số zig-zag đầu

# ========= Header CSV (không label) =========
HEADERS = (
    ["filename"] +
    # Thống kê theo dải
    ["DCT_mean_low","DCT_var_low","DCT_skew_low","DCT_kurt_low",
     "DCT_mean_mid","DCT_var_mid","DCT_skew_mid","DCT_kurt_mid",
     "DCT_mean_high","DCT_var_high","DCT_skew_high","DCT_kurt_high"] +
    # Entropy theo dải
    ["DCT_entropy_low","DCT_entropy_mid","DCT_entropy_high"] +
    # Năng lượng toàn ảnh
    ["DCT_energy_total"] +
    # Zig-zag 20
    [f"DCT_zigzag_{i}" for i in range(ZIGZAG_TAKE)] +
    # Histogram 8 bins theo dải
    [f"DCT_hist_low_bin_{i}" for i in range(HIST_BINS)] +
    [f"DCT_hist_mid_bin_{i}" for i in range(HIST_BINS)] +
    [f"DCT_hist_high_bin_{i}" for i in range(HIST_BINS)]
)

# ========= Tiện ích =========
def list_images(input_dir):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, "**", e), recursive=True))
    files = sorted(files)
    return files

def load_Y_channel(path, target_size=TARGET_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # lấy kênh Y
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img -= 128.0  # level-shift
    return img  # shape (H,W), float32

def iter_blocks(img, block=BLOCK):
    H, W = img.shape
    for y in range(0, H, block):
        for x in range(0, W, block):
            yield y, x, img[y:y+block, x:x+block]

def dct2_block(block_2d):
    # DCT-II 2D = DCT(DCT(block)^T)^T bằng cv2.dct trực tiếp 2D
    return cv2.dct(block_2d)

def zigzag_indices(n=8):
    # Trả về list (row,col) theo thứ tự zig-zag cho ma trận n x n
    idxs = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            r = min(s, n-1)
            c = s - r
            while r >= 0 and c < n:
                idxs.append((r,c))
                r -= 1
                c += 1
        else:
            c = min(s, n-1)
            r = s - c
            while c >= 0 and r < n:
                idxs.append((r,c))
                r += 1
                c -= 1
    return idxs

ZIGZAG_IDXS = zigzag_indices(8)

def band_masks():
    """Trả 3 mask boolean 8x8 cho Low/Mid/High."""
    m_low = np.zeros((8,8), dtype=bool)
    m_low[:LOW_SIZE, :LOW_SIZE] = True  # 3x3 góc trái

    m_mid = np.zeros((8,8), dtype=bool)
    m_mid[:MID_SIZE, :MID_SIZE] = True
    m_mid[m_low] = False  # bỏ phần low

    m_high = ~np.zeros((8,8), dtype=bool)
    m_high[:MID_SIZE, :MID_SIZE] = False  # phần còn lại ngoài 6x6

    return m_low, m_mid, m_high

MASK_LOW, MASK_MID, MASK_HIGH = band_masks()

def safe_stats(x):
    """Tính mean/var/skew/kurt trên mảng 1D x (float). Skew/Kurt fallback nếu thiếu scipy."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=0))
    if HAVE_SCIPY:
        sk = float(scipy_skew(x, bias=False)) if x.size >= 3 else 0.0
        # kurtosis Fisher=False => kurtosis = m4 / s^4 ; Fisher=True => excess
        ku = float(scipy_kurtosis(x, fisher=False, bias=False)) if x.size >= 4 else 0.0
    else:
        # Fallback thủ công
        if var <= 1e-12:
            sk = 0.0
            ku = 3.0
        else:
            std = math.sqrt(var)
            m3 = float(np.mean((x - mean)**3))
            m4 = float(np.mean((x - mean)**4))
            sk = m3 / (std**3 + 1e-12)
            ku = m4 / (std**4 + 1e-12)  # kurtosis (Fisher=False)
    return mean, var, sk, ku

def shannon_entropy(x, bins=ENTROPY_BINS):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    hist, edges = np.histogram(x, bins=bins)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def compute_hist(x, edges):
    # Histogram chuẩn hoá theo edges cố định
    hist, _ = np.histogram(x, bins=edges)
    total = hist.sum()
    if total <= 0:
        return np.zeros(len(edges)-1, dtype=np.float64)
    return (hist / total).astype(np.float64)

def first_pass_minmax(images):
    """Pass 1: ước lượng min/max trên X=log1p(|DCT|) cho 3 dải để tạo bin edges."""
    # (min, max) cho low/mid/high
    mins = [np.inf, np.inf, np.inf]
    maxs = [-np.inf, -np.inf, -np.inf]

    for path in tqdm(images, desc="Pass 1/2: scanning min/max", unit="img"):
        try:
            Y = load_Y_channel(path)
        except Exception:
            continue
        # gom X theo dải trên toàn ảnh
        X_low, X_mid, X_high = [], [], []

        for _, _, blk in iter_blocks(Y):
            if blk.shape != (BLOCK, BLOCK):
                continue
            D = dct2_block(blk)
            A = np.abs(D)
            X = np.log1p(A)  # log1p magnitude

            x_low = X[MASK_LOW]
            x_mid = X[MASK_MID]
            x_high = X[MASK_HIGH]

            X_low.append(x_low)
            X_mid.append(x_mid)
            X_high.append(x_high)

        # concat
        if X_low:
            x = np.concatenate(X_low)
            mins[0] = min(mins[0], float(np.min(x)))
            maxs[0] = max(maxs[0], float(np.max(x)))
        if X_mid:
            x = np.concatenate(X_mid)
            mins[1] = min(mins[1], float(np.min(x)))
            maxs[1] = max(maxs[1], float(np.max(x)))
        if X_high:
            x = np.concatenate(X_high)
            mins[2] = min(mins[2], float(np.min(x)))
            maxs[2] = max(maxs[2], float(np.max(x)))

    # Khử trường hợp nan/inf (nếu ảnh lỗi)
    for i in range(3):
        if not np.isfinite(mins[i]) or not np.isfinite(maxs[i]) or mins[i] >= maxs[i]:
            mins[i], maxs[i] = 0.0, 10.0  # dải mặc định an toàn
    return mins, maxs

def make_equal_edges(low, high, bins=HIST_BINS):
    # edges: [low, ..., high] có (bins) khoảng -> bins+1 mốc
    return np.linspace(low, high, bins+1, dtype=np.float64)

def get_edges_from_dataset(images):
    """Tạo bin edges cho Low/Mid/High; nếu đã có JSON thì load lại."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(BIN_JSON_PATH):
        with open(BIN_JSON_PATH, "r") as f:
            data = json.load(f)
        edges_low = np.array(data["edges_low"], dtype=np.float64)
        edges_mid = np.array(data["edges_mid"], dtype=np.float64)
        edges_high = np.array(data["edges_high"], dtype=np.float64)
        return edges_low, edges_mid, edges_high

    mins, maxs = first_pass_minmax(images)
    # Thêm margin nhỏ để tránh đặt giá trị ở mép
    margins = []
    for mn, mx in zip(mins, maxs):
        span = mx - mn
        margins.append((mn - 0.01*span, mx + 0.01*span))

    (l0, h0), (l1, h1), (l2, h2) = margins
    edges_low = make_equal_edges(l0, h0, HIST_BINS)
    edges_mid = make_equal_edges(l1, h1, HIST_BINS)
    edges_high = make_equal_edges(l2, h2, HIST_BINS)

    # Lưu lại để tái lập
    payload = {
        "edges_low": edges_low.tolist(),
        "edges_mid": edges_mid.tolist(),
        "edges_high": edges_high.tolist(),
        "note": "Histogram edges trên X=log1p(|DCT|), 8 bins, ước lượng từ tập này."
    }
    with open(BIN_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    return edges_low, edges_mid, edges_high

def features_for_image(path, edges_low, edges_mid, edges_high):
    Y = load_Y_channel(path)
    H, W = Y.shape

    # Tích lũy theo dải
    X_low_all, X_mid_all, X_high_all = [], [], []
    energy_total = 0.0

    # Tích lũy zig-zag (trên A), sẽ mean-pool theo block
    zz_vals = [[] for _ in range(ZIGZAG_TAKE)]

    for _, _, blk in iter_blocks(Y):
        if blk.shape != (BLOCK, BLOCK):
            continue
        D = dct2_block(blk)
        A = np.abs(D)
        X = np.log1p(A)

        # Energy tổng trên A (không log)
        energy_total += float(np.sum(A**2))

        # Gom X theo dải
        X_low_all.append(X[MASK_LOW])
        X_mid_all.append(X[MASK_MID])
        X_high_all.append(X[MASK_HIGH])

        # Zig-zag 20 trên A
        for i in range(ZIGZAG_TAKE):
            r, c = ZIGZAG_IDXS[i]
            zz_vals[i].append(float(A[r, c]))

    # Nối dải
    X_low = np.concatenate(X_low_all) if X_low_all else np.array([], dtype=np.float64)
    X_mid = np.concatenate(X_mid_all) if X_mid_all else np.array([], dtype=np.float64)
    X_high = np.concatenate(X_high_all) if X_high_all else np.array([], dtype=np.float64)

    # Thống kê trên X
    mL, vL, sL, kL = safe_stats(X_low)
    mM, vM, sM, kM = safe_stats(X_mid)
    mH, vH, sH, kH = safe_stats(X_high)

    # Entropy trên X
    eL = shannon_entropy(X_low, bins=ENTROPY_BINS)
    eM = shannon_entropy(X_mid, bins=ENTROPY_BINS)
    eH = shannon_entropy(X_high, bins=ENTROPY_BINS)

    # Zig-zag mean-pooling
    zz_feats = []
    for arr in zz_vals:
        if len(arr) == 0:
            zz_feats.append(0.0)
        else:
            zz_feats.append(float(np.mean(arr)))

    # Histogram trên X theo edges cố định (chuẩn hoá)
    hL = compute_hist(X_low, edges_low)
    hM = compute_hist(X_mid, edges_mid)
    hH = compute_hist(X_high, edges_high)

    # Kết quả theo đúng HEADERS
    row = [os.path.relpath(path, INPUT_DIR)]
    row += [mL, vL, sL, kL, mM, vM, sM, kM, mH, vH, sH, kH]
    row += [eL, eM, eH]
    row += [energy_total]
    row += zz_feats
    row += hL.tolist() + hM.tolist() + hH.tolist()
    return row

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = list_images(INPUT_DIR)
    if len(images) == 0:
        print(f"No images found in: {INPUT_DIR}")
        return

    # Tạo edges (load nếu đã có)
    edges_low, edges_mid, edges_high = get_edges_from_dataset(images)

    # Pass 2: tính đặc trưng và ghi CSV
    rows = []
    for path in tqdm(images, desc="Pass 2/2: extracting features", unit="img"):
        try:
            row = features_for_image(path, edges_low, edges_mid, edges_high)
            rows.append(row)
        except Exception as ex:
            print(f"Skip {path} due to error: {ex}")

    df = pd.DataFrame(rows, columns=HEADERS)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Done. Wrote {len(df)} rows to: {CSV_PATH}")
    print(f"Saved histogram bin edges at: {BIN_JSON_PATH}")

if __name__ == "__main__":
    main()
