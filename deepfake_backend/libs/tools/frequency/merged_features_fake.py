import pandas as pd

BASE = "/Applications/Tien/deepfake/extract-celeb"
srm = pd.read_csv(f"{BASE}/SRM/srm_features_fake.csv")
dct = pd.read_csv(f"{BASE}/DCT/dct_features_fake.csv")
fft = pd.read_csv(f"{BASE}/FFT/fft_features_fake.csv")

merged_fake = srm.merge(dct, on="filename", how="inner").merge(fft, on="filename", how="inner")
print("✅ Merged FAKE shape:", merged_fake.shape, "| NaN:", merged_fake.isna().sum().sum())

out_fake = f"{BASE}/merged_features_fake.csv"
merged_fake.to_csv(out_fake, index=False)
print("💾 Saved:", out_fake)
