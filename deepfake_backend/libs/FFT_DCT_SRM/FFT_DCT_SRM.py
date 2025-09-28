# ================================
# DEEPFAKE DETECTION PIPELINE (NO-VISUALIZE)
# FFT + DCT + SRM + MERGE | Save matrices & images, print progress
# ================================

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import os
import subprocess
import sys

# --------- Install (safe) ----------
def install_packages():
    pkgs = ['opencv-python', 'numpy', 'scipy', 'scikit-learn']
    for p in pkgs:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', p])
        except Exception:
            pass

print("Installing required packages (idempotent)...")
install_packages()

# ================================
# Step 2: Image I/O & Preprocess
# ================================
def load_existing_image(image_path):
    print(f"[Load] {image_path}")
    if not os.path.exists(image_path):
        print(f"  ! File not found")
        return None, None
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("  ! Cannot read image")
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"  OK shape={img_rgb.shape}")
    return img_rgb, image_path

def preprocess_image(img_array):
    # RGBA -> RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Gray or RGB
    if len(img_array.shape) == 2:
        img_gray = img_array
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        if img_array.shape[2] == 3:
            img_rgb = img_array
            # luminance
            img_gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        else:
            raise ValueError("Unsupported image format")
    img_gray = img_gray.astype(np.uint8)
    print(f"[Preprocess] gray={img_gray.shape}")
    return img_rgb, img_gray

# ================================
# Step 3: FFT
# ================================
def apply_fft_analysis(img_gray):
    print("[FFT] start")
    fft_result = fft2(img_gray)
    fft_shifted = fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)
    phase_spectrum = np.angle(fft_shifted)
    log_magnitude = np.log(magnitude_spectrum + 1.0)
    # display image
    mag_disp = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-12)
    mag_disp = (mag_disp * 255).astype(np.uint8)
    print(f"  done shape={magnitude_spectrum.shape}")
    return {
        'original_fft': fft_result,
        'fft_shifted': fft_shifted,
        'magnitude_spectrum': magnitude_spectrum,
        'phase_spectrum': phase_spectrum,
        'log_magnitude': log_magnitude,
        'magnitude_display': mag_disp
    }

def extract_fft_features(fft_results):
    magnitude = fft_results['magnitude_spectrum']
    h, w = magnitude.shape
    ch, cw = h//2, w//2
    total_energy = float(np.sum(magnitude))
    low = magnitude[ch-20:ch+20, cw-20:cw+20]
    low_energy = float(np.sum(low))
    high_energy = float(total_energy - low_energy)
    ratio = high_energy / (low_energy + 1e-8)
    y, x = np.mgrid[:h, :w]
    cx = float(np.sum(x * magnitude) / (total_energy + 1e-12))
    cy = float(np.sum(y * magnitude) / (total_energy + 1e-12))
    rr = np.sqrt((x - cw)**2 + (y - ch)**2).astype(int)
    tbin = np.bincount(rr.ravel(), magnitude.ravel())
    nr = np.bincount(rr.ravel())
    radial = tbin / (nr + 1e-8)
    peaks, _ = find_peaks(radial, height=np.mean(radial))
    return {
        'total_energy': total_energy,
        'low_freq_energy': low_energy,
        'high_freq_energy': high_energy,
        'energy_ratio': ratio,
        'spectral_centroid': (cx, cy),
        'num_peaks': int(len(peaks)),
        'radial_profile': radial
    }

def save_fft_outputs(fft_results, prefix="fft"):
    # matrices
    np.savetxt(f"{prefix}_magnitude_spectrum.txt", fft_results['magnitude_spectrum'], fmt="%.6f")
    np.save(f"{prefix}_magnitude_spectrum.npy", fft_results['magnitude_spectrum'])
    # images
    cv2.imwrite(f"{prefix}_magnitude_display.png", fft_results['magnitude_display'])
    print(f"[FFT] saved: {prefix}_magnitude_spectrum.txt/.npy & {prefix}_magnitude_display.png")

# ================================
# Step 4: DCT
# ================================
def apply_dct_analysis(img_gray, block_size=8):
    print("[DCT] start")
    h, w = img_gray.shape
    hb = (h // block_size) * block_size
    wb = (w // block_size) * block_size
    crop = img_gray[:hb, :wb]
    dc, ac, blocks = [], [], []
    for i in range(0, hb, block_size):
        for j in range(0, wb, block_size):
            block = crop[i:i+block_size, j:j+block_size].astype(np.float32)
            dctb = cv2.dct(block)
            dc.append(dctb[0,0])
            tmp = dctb.copy(); tmp[0,0] = 0.0
            ac.extend(tmp.flatten())
            blocks.append(dctb)
    nb_h, nb_w = hb // block_size, wb // block_size
    dct_matrix = np.zeros((hb, wb), dtype=np.float32)
    idx = 0
    for r in range(nb_h):
        for c in range(nb_w):
            si, sj = r*block_size, c*block_size
            dct_matrix[si:si+block_size, sj:sj+block_size] = blocks[idx]
            idx += 1
    dct_log = np.log(np.abs(dct_matrix) + 1.0)
    dct_disp = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-12)
    dct_disp = (dct_disp * 255).astype(np.uint8)
    print(f"  done shape={dct_matrix.shape}, blocks={len(blocks)}")
    return {
        'dct_matrix': dct_matrix,
        'dc_coefficients': np.array(dc, dtype=np.float32),
        'ac_coefficients': np.array(ac, dtype=np.float32),
        'dct_log': dct_log,
        'dct_display': dct_disp,
        'dct_blocks': np.array(blocks, dtype=np.float32)
    }

def extract_dct_features(dct_results):
    dc = dct_results['dc_coefficients']
    ac = dct_results['ac_coefficients']
    ac_nz = ac[ac != 0]
    return {
        'dc_mean': float(np.mean(dc)),
        'dc_std': float(np.std(dc)),
        'dc_variance': float(np.var(dc)),
        'ac_energy': float(np.sum(ac_nz**2)) if ac_nz.size else 0.0,
        'ac_sparsity': float(ac_nz.size) / float(ac.size) if ac.size else 0.0,
        'dc_range': float(np.max(dc) - np.min(dc)) if dc.size else 0.0
    }

def save_dct_outputs(dct_results, prefix="dct"):
    np.savetxt(f"{prefix}_matrix.txt", dct_results['dct_matrix'], fmt="%.6f")
    np.save(f"{prefix}_matrix.npy", dct_results['dct_matrix'])
    cv2.imwrite(f"{prefix}_display.png", dct_results['dct_display'])
    print(f"[DCT] saved: {prefix}_matrix.txt/.npy & {prefix}_display.png")

# ================================
# Step 5: SRM
# ================================
def create_srm_filters():
    f = {}
    f['hpf_1'] = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], np.float32)
    f['hpf_2'] = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
    f['edge_h'] = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
    f['edge_v'] = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
    f['diag_1'] = np.array([[-1,0,1],[0,0,0],[1,0,-1]], np.float32)
    f['diag_2'] = np.array([[1,0,-1],[0,0,0],[-1,0,1]], np.float32)
    f['laplacian_4'] = np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)
    f['laplacian_8'] = np.array([[1,1,1],[1,-8,1],[1,1,1]], np.float32)
    f['noise_1'] = np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,-4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]], np.float32)
    f['noise_2'] = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], np.float32)
    return f

def apply_srm_analysis(img_gray):
    print("[SRM] start")
    filters = create_srm_filters()
    filt_resp = {}
    combined = np.zeros_like(img_gray, dtype=np.float32)
    for name, ker in filters.items():
        resp = cv2.filter2D(img_gray.astype(np.float32), -1, ker)
        filt_resp[name] = resp
        combined += np.abs(resp) * 0.1
    combined_norm = combined / (combined.max() + 1e-12) if combined.max() > 0 else combined
    srm_disp = (combined_norm * 255).astype(np.uint8)
    srm_log = np.log(combined + 1.0)
    srm_log_disp = ((srm_log - srm_log.min()) / (srm_log.max() - srm_log.min() + 1e-12) * 255).astype(np.uint8)
    print(f"  done shape={combined.shape}")
    return {
        'filter_responses': filt_resp,
        'combined_response': combined,
        'srm_matrix': combined_norm,
        'srm_display': srm_disp,
        'srm_log': srm_log,
        'srm_log_display': srm_log_disp
    }

def extract_srm_features(srm_results):
    c = srm_results['combined_response']
    feat = {
        'srm_mean': float(np.mean(c)),
        'srm_std': float(np.std(c)),
        'srm_variance': float(np.var(c)),
        'srm_energy': float(np.sum(c**2)),
        'srm_sparsity': float(np.sum(c==0))/float(c.size)
    }
    thr = np.percentile(c, 90)
    feat['edge_density'] = float(np.sum(c > thr))/float(c.size)
    h, w = c.shape
    center = c[1:h-1, 1:w-1]
    neighbors = [c[0:h-2,1:w-1], c[2:h,1:w-1], c[1:h-1,0:w-2], c[1:h-1,2:w]]
    tex = 0.0
    for nb in neighbors:
        tex += float(np.sum(np.abs(center - nb)))
    feat['texture_complexity'] = tex / (4.0 * center.size + 1e-12)
    mad = float(np.median(np.abs(c - np.median(c))))
    feat['noise_level'] = 1.4826 * mad
    hm, wm = h//2, w//2
    quads = [c[:hm,:wm], c[:hm,wm:], c[hm:,:wm], c[hm:,wm:]]
    qE = [float(np.sum(q**2)) for q in quads]
    feat['spatial_uniformity'] = float(np.std(qE)) / (float(np.mean(qE)) + 1e-12)
    return feat

def save_srm_outputs(srm_results, prefix="srm"):
    np.savetxt(f"{prefix}_matrix.txt", srm_results['srm_matrix'], fmt="%.6f")
    np.save(f"{prefix}_matrix.npy", srm_results['srm_matrix'])
    cv2.imwrite(f"{prefix}_display.png", srm_results['srm_display'])
    cv2.imwrite(f"{prefix}_log_display.png", srm_results['srm_log_display'])
    print(f"[SRM] saved: {prefix}_matrix.txt/.npy & {prefix}_display.png, {prefix}_log_display.png")

# ================================
# Step 6: Merge
# ================================
def normalize_matrix(matrix, method='minmax', target_range=(0,1)):
    shp = matrix.shape
    flat = matrix.flatten().reshape(-1,1)
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=target_range)
        norm = scaler.fit_transform(flat)
    elif method == 'zscore':
        mean = np.mean(flat); std = np.std(flat)
        norm = (flat - mean) / (std + 1e-8)
        lo, hi = target_range
        norm = (norm - norm.min())/(norm.max()-norm.min()+1e-12)*(hi-lo)+lo
    else:
        med = np.median(flat)
        q25, q75 = np.percentile(flat,25), np.percentile(flat,75)
        iqr = q75-q25
        norm = (flat - med)/(iqr + 1e-8)
        lo, hi = target_range
        norm = (norm - norm.min())/(norm.max()-norm.min()+1e-12)*(hi-lo)+lo
    return norm.reshape(shp)

def resize_matrix_to_target(matrix, target_size=(224,224)):
    if matrix.shape[:2] == target_size:
        return matrix
    return cv2.resize(matrix.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

def analyze_matrix_statistics(matrix, name):
    stats = {
        'shape': matrix.shape,
        'dtype': str(matrix.dtype),
        'min': float(np.min(matrix)),
        'max': float(np.max(matrix)),
        'mean': float(np.mean(matrix)),
        'std': float(np.std(matrix)),
        'median': float(np.median(matrix)),
        'range': float(np.max(matrix)-np.min(matrix)),
        'sparsity': float(np.sum(matrix==0))/float(matrix.size),
        'p25': float(np.percentile(matrix,25)),
        'p75': float(np.percentile(matrix,75)),
    }
    print(f"[Stats] {name}: shape={stats['shape']} range=[{stats['min']:.6f},{stats['max']:.6f}] mean±std={stats['mean']:.6f}±{stats['std']:.6f}")
    return stats

def merge_frequency_matrices(fft_results, dct_results, srm_results,
                             target_size=(224,224), normalization_method='minmax'):
    print("[Merge] begin")
    fft_mat = fft_results['magnitude_spectrum']
    dct_mat = dct_results['dct_matrix']
    srm_mat = srm_results['srm_matrix']
    analyze_matrix_statistics(fft_mat, "FFT raw")
    analyze_matrix_statistics(dct_mat, "DCT raw")
    analyze_matrix_statistics(srm_mat, "SRM raw")

    print(f"[Merge] resize-> {target_size}")
    fft_r = resize_matrix_to_target(fft_mat, target_size)
    dct_r = resize_matrix_to_target(dct_mat, target_size)
    srm_r = resize_matrix_to_target(srm_mat, target_size)
    print("  resized shapes:", fft_r.shape, dct_r.shape, srm_r.shape)

    print(f"[Merge] normalize ({normalization_method})")
    fft_n = normalize_matrix(fft_r, normalization_method, (0,1))
    dct_n = normalize_matrix(dct_r, normalization_method, (0,1))
    srm_n = normalize_matrix(srm_r, normalization_method, (0,1))
    analyze_matrix_statistics(fft_n, "FFT norm")
    analyze_matrix_statistics(dct_n, "DCT norm")
    analyze_matrix_statistics(srm_n, "SRM norm")

    tensor = np.stack([fft_n, dct_n, srm_n], axis=2).astype(np.float32)
    tensor_u8 = (tensor * 255).astype(np.uint8)
    merged_gray = np.mean(tensor, axis=2)
    weights = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    merged_weighted = np.average(tensor, axis=2, weights=weights)

    print(f"[Merge] tensor shape={tensor.shape} dtype={tensor.dtype} range=[{tensor.min():.6f},{tensor.max():.6f}]")
    return {
        'merged_tensor': tensor,
        'merged_uint8': tensor_u8,
        'merged_grayscale': merged_gray,
        'merged_weighted': merged_weighted,
        'fft_normalized': fft_n,
        'dct_normalized': dct_n,
        'srm_normalized': srm_n,
        'target_size': target_size,
        'normalization_method': normalization_method,
        'weights_used': weights
    }

def save_merged_outputs(merged, prefix="merged"):
    np.save(f"{prefix}_tensor.npy", merged['merged_tensor'])
    np.savetxt(f"{prefix}_tensor_flat.txt", merged['merged_tensor'].reshape(-1,3), fmt="%.6f")  # optional
    # save merged rgb + channels + grayscale/weighted
    cv2.imwrite(f"{prefix}_rgb.png", cv2.cvtColor(merged['merged_uint8'], cv2.COLOR_RGB2BGR))
    for i, name in enumerate(['fft','dct','srm']):
        ch = (merged['merged_tensor'][:,:,i] * 255).astype(np.uint8)
        cv2.imwrite(f"{prefix}_{name}_channel.png", ch)
        np.savetxt(f"{prefix}_{name}_normalized.txt", merged['merged_tensor'][:,:,i], fmt="%.6f")
        np.save(f"{prefix}_{name}_normalized.npy", merged['merged_tensor'][:,:,i])
    gray_u8 = (merged['merged_grayscale']*255).astype(np.uint8)
    w_u8 = (merged['merged_weighted']*255).astype(np.uint8)
    cv2.imwrite(f"{prefix}_grayscale.png", gray_u8)
    cv2.imwrite(f"{prefix}_weighted.png", w_u8)
    print(f"[Merge] saved: {prefix}_tensor.npy, {prefix}_rgb.png, channels & grayscale/weighted")

# ================================
# Step 7: (Optional) Text-merger helpers — giữ nguyên để bạn dùng khi cần
# ================================
def load_matrix_from_txt(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"! not found: {file_path}")
            return None
        m = np.loadtxt(file_path)
        print(f"[LoadTxt] {file_path} shape={m.shape} range=[{m.min():.6f},{m.max():.6f}]")
        return m
    except Exception as e:
        print(f"[LoadTxt] error {file_path}: {e}")
        return None

def resize_matrix_if_needed(matrix, target_shape):
    if matrix.shape == target_shape: return matrix
    r = cv2.resize(matrix.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    print(f"[ResizeTxt] {matrix.shape} -> {r.shape}")
    return r

def normalize_matrix_to_range(matrix, target_min=0, target_max=1):
    mn, mx = matrix.min(), matrix.max()
    if mx == mn:
        return np.full_like(matrix, (target_min+target_max)/2.0, dtype=np.float32)
    norm = (matrix - mn)/(mx - mn)
    return (norm*(target_max-target_min) + target_min).astype(np.float32)

def merge_three_matrices_weighted_average(m1,m2,m3,weights=[0.33,0.34,0.33]):
    w = np.array(weights, dtype=np.float32); w = w / w.sum()
    target_shape = (max(m1.shape[0],m2.shape[0],m3.shape[0]), max(m1.shape[1],m2.shape[1],m3.shape[1]))
    m1r = resize_matrix_if_needed(m1, target_shape)
    m2r = resize_matrix_if_needed(m2, target_shape)
    m3r = resize_matrix_if_needed(m3, target_shape)
    return w[0]*m1r + w[1]*m2r + w[2]*m3r

def merge_three_matrices_layer_stack(m1,m2,m3):
    target_shape = (max(m1.shape[0],m2.shape[0],m3.shape[0]), max(m1.shape[1],m2.shape[1],m3.shape[1]))
    m1r = resize_matrix_if_needed(m1, target_shape)
    m2r = resize_matrix_if_needed(m2, target_shape)
    m3r = resize_matrix_if_needed(m3, target_shape)
    return np.stack([m1r,m2r,m3r], axis=2)

def save_matrix_to_txt(matrix, filename, fmt='%.6f'):
    if matrix.ndim == 2:
        np.savetxt(filename, matrix, fmt=fmt)
        print(f"[SaveTxt] {filename} shape={matrix.shape}")
    else:
        # save each layer
        base = filename.replace('.txt','')
        for i in range(matrix.shape[2]):
            np.savetxt(f"{base}_layer{i+1}.txt", matrix[:,:,i], fmt=fmt)
        print(f"[SaveTxt] 3D -> layers saved with base {base}_layer*.txt")

def merge_matrices_from_files(file1, file2, file3, output_filename="merged_matrix.txt",
                              merge_method="weighted_average", normalize_before_merge=True, weights=[0.33,0.34,0.33]):
    print("[TxtMerge] begin")
    m1, m2, m3 = load_matrix_from_txt(file1), load_matrix_from_txt(file2), load_matrix_from_txt(file3)
    if any(m is None for m in [m1,m2,m3]):
        print("  ! load fail")
        return False
    if normalize_before_merge:
        m1 = normalize_matrix_to_range(m1,0,1)
        m2 = normalize_matrix_to_range(m2,0,1)
        m3 = normalize_matrix_to_range(m3,0,1)
    if merge_method == "layer_stack":
        merged = merge_three_matrices_layer_stack(m1,m2,m3)
    else:
        merged = merge_three_matrices_weighted_average(m1,m2,m3,weights)
    save_matrix_to_txt(merged, output_filename)
    print("[TxtMerge] done")
    return True

# ================================
# Step 8: Main – no matplotlib, chỉ in log + lưu file
# ================================
def main():
    print("=== START PIPELINE (NO-VISUALIZE) ===")
    image_path = "/content/000_id0_0000_frame_0000.jpg"

    # 1) Load & preprocess
    img_rgb, _ = load_existing_image(image_path)
    if img_rgb is None:
        return
    img_rgb, img_gray = preprocess_image(img_rgb)
    cv2.imwrite("original_rgb.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print("[Save] original_rgb.png")

    # 2) FFT
    fft = apply_fft_analysis(img_gray)
    ffeat = extract_fft_features(fft)
    print(f"[FFT] Energy Ratio={ffeat['energy_ratio']:.4f}, Peaks={ffeat['num_peaks']}")
    save_fft_outputs(fft, prefix="fft")

    # 3) DCT
    dct = apply_dct_analysis(img_gray)
    dfeat = extract_dct_features(dct)
    print(f"[DCT] DC mean={dfeat['dc_mean']:.2f}, AC sparsity={dfeat['ac_sparsity']:.4f}")
    save_dct_outputs(dct, prefix="dct")

    # 4) SRM
    srm = apply_srm_analysis(img_gray)
    sfeat = extract_srm_features(srm)
    print(f"[SRM] mean={sfeat['srm_mean']:.6f}, edge_density={sfeat['edge_density']:.4f}")
    save_srm_outputs(srm, prefix="srm")

    # 5) Merge
    merged = merge_frequency_matrices(fft, dct, srm, target_size=(224,224), normalization_method='minmax')
    save_merged_outputs(merged, prefix="frequency_merged")

    # 6) Quick text merge (nếu đã có .txt)
    if os.path.exists("srm_matrix.txt") and os.path.exists("fft_magnitude_spectrum.txt"):
        print("[TxtMerge] example SRM + Magnitude + SRM")
        merge_matrices_from_files("srm_matrix.txt","fft_magnitude_spectrum.txt","srm_matrix.txt",
                                  output_filename="merged_srm_mag_srm.txt",
                                  merge_method="weighted_average",
                                  normalize_before_merge=True, weights=[0.3,0.4,0.3])

    print("=== DONE ===")
    print("Artifacts:")
    print(" - fft_magnitude_spectrum.txt/.npy & fft_magnitude_display.png")
    print(" - dct_matrix.txt/.npy & dct_display.png")
    print(" - srm_matrix.txt/.npy & srm_display.png, srm_log_display.png")
    print(" - frequency_merged_tensor.npy & frequency_merged_rgb.png (+channels/grayscale/weighted)")

if __name__ == "__main__":
    main()
