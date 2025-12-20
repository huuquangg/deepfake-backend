"""
Batch OpenFace Feature Extraction API
Extracts features from up to 30 frames, returns CSV data, and cleans up files
"""
import os
import subprocess
import shutil
import glob
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Batch OpenFace Feature Extraction API",
    description="Extract OpenFace features from up to 30 frames and return CSV data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENFACE_BINARY = os.getenv("OPENFACE_BINARY", "/usr/local/bin/FeatureExtraction")
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/openface_batch")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/batch_output")
BATCH_SIZE = 30

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def of_fixed_columns() -> List[str]:
    """
    Define the 674 fixed OpenFace feature columns
    
    Returns:
        List of column names for the 674 features
    """
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


def read_csv_data(csv_path: str, frame_index: int = -1) -> Dict[str, Any]:
    """
    Read OpenFace CSV output and filter to 674 fixed features
    
    Args:
        csv_path: Path to CSV file
        frame_index: 0-based frame index for alignment (optional)
    
    Returns:
        Dictionary containing headers, frame_index, and row data (filtered to 674 features)
    """
    try:
        # Read CSV - OpenFace uses space or comma separator
        df = pd.read_csv(csv_path, sep=None, engine='python')
        
        # Get the 674 fixed columns
        fixed_cols = of_fixed_columns()
        
        # Strip whitespace from column names for matching
        df.columns = df.columns.str.strip()
        
        # Filter dataframe to only include columns that exist in both df and fixed_cols
        # Preserve order from fixed_cols
        available_cols = [col for col in fixed_cols if col in df.columns]
        
        if not available_cols:
            logger.warning(f"No matching columns found in {csv_path}. Available columns: {df.columns.tolist()[:10]}...")
            df_filtered = df  # Fallback to original if no matches
        else:
            df_filtered = df[available_cols]
            logger.info(f"Filtered {len(df.columns)} columns down to {len(available_cols)} columns")
        
        # Convert to dictionary format
        data = {
            "filename": os.path.basename(csv_path),
            "frame_index": frame_index,  # 0-based frame index for alignment
            "headers": df_filtered.columns.tolist(),
            "num_rows": len(df_filtered),
            "num_columns": len(df_filtered.columns),
            "data": df_filtered.to_dict(orient='records')  # List of dicts, one per row
        }
        
        logger.info(f"Read CSV: {data['num_rows']} rows, {data['num_columns']} columns (filtered from {len(df.columns)}) from {data['filename']}")
        return data
        
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        raise


@app.get("/")
def read_root():
    return {
        "message": "Batch OpenFace Feature Extraction API 🚀",
        "version": "1.0.0",
        "batch_size": BATCH_SIZE,
        "endpoints": {
            "health": "/health",
            "extract_batch": "/extract/batch"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    openface_exists = os.path.exists(OPENFACE_BINARY)
    
    return {
        "status": "healthy" if openface_exists else "unhealthy",
        "openface_binary": OPENFACE_BINARY,
        "openface_available": openface_exists,
        "temp_dir": TEMP_DIR,
        "output_dir": OUTPUT_DIR,
        "batch_size": BATCH_SIZE
    }


@app.post("/extract/batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    cleanup: bool = Form(True)
):
    """
    Extract OpenFace features from up to 30 frames
    
    Pipeline:
    1. Upload frames (up to 30)
    2. Run OpenFace feature extraction
    3. Read CSV data
    4. Return response with all features
    5. Clean up CSV files (if cleanup=True)
    
    Args:
        files: List of frame image files (max 30)
        session_id: Optional session ID
        cleanup: Whether to delete CSV files after processing (default: True)
    
    Returns:
        JSON with CSV data for each frame
    """
    
    # Validate batch size
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit. Max {BATCH_SIZE} frames, got {len(files)}"
        )
    
    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File has no filename")
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    # Generate session ID
    request_id = session_id or f"batch_{os.urandom(8).hex()}"
    
    # Create temporary directories
    temp_frames_dir = os.path.join(TEMP_DIR, request_id)
    output_subdir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(temp_frames_dir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)
    
    csv_files = []
    
    try:
        # Step 1: Save uploaded frames
        logger.info(f"[{request_id}] Saving {len(files)} frames...")
        saved_files = []
        for idx, file in enumerate(files):
            filename = f"frame_{idx:04d}{Path(file.filename).suffix}"
            file_path = os.path.join(temp_frames_dir, filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_files.append(file_path)
        
        logger.info(f"[{request_id}] Saved {len(saved_files)} frames to {temp_frames_dir}")
        
        # Step 2: Run OpenFace feature extraction
        logger.info(f"[{request_id}] Running OpenFace extraction...")
        
        cmd = [
            OPENFACE_BINARY,
            "-fdir", temp_frames_dir,
            "-out_dir", output_subdir,
            "-2Dfp",  # 2D facial landmarks
            "-3Dfp",  # 3D facial landmarks
            "-pdmparams",  # PDM parameters
            "-pose",  # Head pose
            "-aus",  # Action Units
            "-gaze",  # Gaze estimation
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=180  # 3 minutes for batch
        )
        
        if result.returncode != 0:
            logger.error(f"[{request_id}] OpenFace error: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenFace extraction failed: {result.stderr}"
            )
        
        # Step 3: Find generated CSV files
        csv_pattern = os.path.join(output_subdir, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise HTTPException(
                status_code=500,
                detail="No CSV files generated by OpenFace"
            )
        
        logger.info(f"[{request_id}] Generated {len(csv_files)} CSV files")
        
        # Step 4: Read CSV data from all files
        csv_data_list = []
        
        for idx, csv_file in enumerate(sorted(csv_files)):
            try:
                csv_data = read_csv_data(csv_file, frame_index=idx)
                csv_data_list.append(csv_data)
                    
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to read CSV {csv_file}: {e}")
        
        if not csv_data_list:
            raise HTTPException(
                status_code=500,
                detail="No CSV data extracted"
            )
        
        logger.info(f"[{request_id}] Read {len(csv_data_list)} CSV files")
        
        # Step 5: Cleanup CSV files if requested
        if cleanup:
            try:
                for csv_file in csv_files:
                    os.remove(csv_file)
                cleanup_status = f"Deleted {len(csv_files)} CSV files"
                logger.info(f"[{request_id}] Cleanup: {cleanup_status}")
            except Exception as e:
                cleanup_status = f"Cleanup failed: {str(e)}"
                logger.warning(f"[{request_id}] {cleanup_status}")
        else:
            cleanup_status = f"CSV files preserved at {output_subdir}"
        
        # Step 6: Calculate summary statistics
        total_rows = sum(csv_data['num_rows'] for csv_data in csv_data_list)
        total_features = csv_data_list[0]['num_columns'] if csv_data_list else 0
        
        # Step 7: Return response
        response = {
            "status": "success",
            "session_id": request_id,
            "frames_uploaded": len(saved_files),
            "csv_files_generated": len(csv_data_list),
            "csv_data": csv_data_list,
            "summary": {
                "total_csv_files": len(csv_data_list),
                "total_data_rows": total_rows,
                "features_per_row": total_features,
                "cleanup_performed": cleanup,
                "output_directory": output_subdir if not cleanup else None
            },
            "cleanup_status": cleanup_status
        }
        
        return response
        
    except subprocess.TimeoutExpired:
        logger.error(f"[{request_id}] OpenFace timeout")
        raise HTTPException(status_code=504, detail="Processing timeout (3min limit)")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Always cleanup temporary frame files
        if os.path.exists(temp_frames_dir):
            try:
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                logger.info(f"[{request_id}] Cleaned up temporary frame directory")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to cleanup temp dir: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Batch OpenFace Feature Extraction API on {host}:{port}")
    logger.info(f"Batch size: {BATCH_SIZE} frames")
    uvicorn.run(app, host=host, port=port)
