import pandas as pd

BASE = "/Applications/Tien/deepfake/extract-celeb"

srm = pd.read_csv(f"{BASE}/SRM/srm_features_real.csv")
dct = pd.read_csv(f"{BASE}/DCT/dct_features_real.csv")
fft = pd.read_csv(f"{BASE}/FFT/fft_features_real.csv")

merged = srm.merge(dct, on="filename", how="inner").merge(fft, on="filename", how="inner")
print("✅ Merged REAL shape:", merged.shape)
print("🚫 NaN:", merged.isna().sum().sum())

out_real = f"{BASE}/merged_features_real.csv"
merged.to_csv(out_real, index=False)
print("💾 Saved:", out_real)
