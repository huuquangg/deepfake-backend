# Frequency-Domain Feature Extraction API

Batch API for extracting frequency domain features (FFT, DCT, SRM) from image frames.

## Features

Extracts **283 frequency domain features** per frame:

- **SRM (Steganalysis Rich Model)**: 120 features
  - 20 high-pass filter kernels
  - 6 statistics per kernel (mean, variance, skewness, kurtosis, entropy, energy)

- **DCT (Discrete Cosine Transform)**: 57 features
  - Band statistics (low, mid, high frequency)
  - Zigzag pattern coefficients (first 20)
  - Histogram features (8 bins × 3 bands)

- **FFT (Fast Fourier Transform)**: 103 features
  - Power spectrum distribution
  - Radial and angular profiles
  - Spectral shape metrics (flatness, entropy, rolloff)
  - JPEG artifact detection

## Installation

```bash
cd /home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/frequency-extraction
pip install -r requirements.txt
```

## Running the Service

```bash
python batch_api.py
```

Default configuration:
- Host: `0.0.0.0`
- Port: `8002`
- Batch size: 30 frames

Environment variables:
```bash
export PORT=8002
export HOST=0.0.0.0
export TEMP_DIR=/tmp/frequency_batch
```

## API Endpoints

### 1. Health Check

```bash
GET http://localhost:8002/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "frequency-extraction",
  "temp_dir": "/tmp/frequency_batch",
  "batch_size": 30,
  "target_size": [256, 256],
  "features": {
    "SRM_kernels": 20,
    "DCT_features": 57,
    "FFT_features": 103,
    "total_features": 283
  }
}
```

### 2. Extract Batch Features

```bash
POST http://localhost:8002/extract/batch
```

**Request (form-data):**
- `files`: Image files (JPG, PNG, BMP) - max 30 frames
- `session_id` (optional): Session identifier
- `cleanup` (optional): Delete temp files after processing (default: true)

**Example using curl:**
```bash
curl -X POST http://localhost:8002/extract/batch \
  -F "files=@frame_0001.jpg" \
  -F "files=@frame_0002.jpg" \
  -F "session_id=test_session" \
  -F "cleanup=true"
```

**Example using Python:**
```python
import requests

files = [
    ("files", open("frame_0001.jpg", "rb")),
    ("files", open("frame_0002.jpg", "rb"))
]

data = {
    "session_id": "test_session",
    "cleanup": "true"
}

response = requests.post(
    "http://localhost:8002/extract/batch",
    files=files,
    data=data
)

result = response.json()
print(f"Extracted features from {result['num_frames']} frames")
print(f"Total features per frame: {result['csv_data']['num_columns']}")
```

**Response Format:**
```json
{
  "request_id": "freq_a1b2c3d4e5f6g7h8",
  "num_frames": 2,
  "csv_data": {
    "filename": "freq_a1b2c3d4e5f6g7h8_frequency_features.csv",
    "headers": ["filename", "SRM_mean_1", "SRM_var_1", ..., "do_hann"],
    "num_rows": 2,
    "num_columns": 283,
    "data": [
      {
        "filename": "frame_0000.jpg",
        "SRM_mean_1": -0.0001234,
        "SRM_var_1": 0.5678,
        ...
        "fft_jpeg_8x8_diag": 3.4567,
        "width": 1920,
        "height": 1080,
        "color_mode": "gray",
        "resize_to": 256,
        "do_hann": 1
      },
      {
        "filename": "frame_0001.jpg",
        ...
      }
    ]
  },
  "metadata": {
    "target_size": [256, 256],
    "features": {
      "SRM": 120,
      "DCT": 57,
      "FFT": 103,
      "total": 283
    }
  }
}
```

## Feature Columns

The response includes 283 features in this order:

1. **filename** (1 column)
2. **SRM features** (120 columns): `SRM_mean_1` through `SRM_energy_20`
3. **DCT features** (57 columns):
   - Band statistics: `DCT_mean_low`, `DCT_var_low`, etc.
   - Zigzag: `DCT_zigzag_0` through `DCT_zigzag_19`
   - Histograms: `DCT_hist_low_bin_0` through `DCT_hist_high_bin_7`
4. **FFT features** (103 columns):
   - Global: `fft_psd_total`, `fft_E_low`, etc.
   - Angular: `fft_aps_0` through `fft_aps_11`
   - Radial: `fft_rps_0` through `fft_rps_63`
   - Peaks: `fft_peak1_r`, `fft_peak1_val`, etc.
   - JPEG: `fft_jpeg_8x8_x`, `fft_jpeg_8x8_y`, `fft_jpeg_8x8_diag`
5. **Metadata** (5 columns): `width`, `height`, `color_mode`, `resize_to`, `do_hann`

## Integration with Video Streaming

This service follows the same API pattern as `openface-extraction` for seamless integration:

```go
// In aggregator.go, add frequency extraction
resp, err := http.Post(
    a.config.FrequencyAPIURL + "/extract/batch",
    "multipart/form-data",
    body,
)

// Response format matches OpenFace API structure
var result struct {
    RequestID string `json:"request_id"`
    NumFrames int    `json:"num_frames"`
    CSVData   struct {
        Filename   string                   `json:"filename"`
        Headers    []string                 `json:"headers"`
        NumRows    int                      `json:"num_rows"`
        NumColumns int                      `json:"num_columns"`
        Data       []map[string]interface{} `json:"data"`
    } `json:"csv_data"`
}
```

## Technical Details

- **Image Processing**: All frames resized to 256×256 grayscale
- **SRM**: 20 high-pass kernels with statistical analysis
- **DCT**: Frequency band decomposition with zigzag scanning
- **FFT**: 2D power spectrum with Hanning window, DC removal
- **Performance**: ~50-100ms per frame (single-threaded)

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Invalid request (wrong file type, too many files, etc.)
- `500`: Internal server error

## Cleanup

By default, temporary files are automatically deleted after processing. To keep files for debugging:

```bash
curl -X POST http://localhost:8002/extract/batch \
  -F "files=@frame.jpg" \
  -F "cleanup=false"
```

Temporary files are stored in: `/tmp/frequency_batch/<session_id>/`
