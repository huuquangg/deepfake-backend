"""
OpenFace Feature Extraction API
Minimal API for extracting facial features using OpenFace
"""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenFace Feature Extraction API",
    description="API for extracting facial features from images and videos using OpenFace",
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
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/openface")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/output")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {
        "message": "OpenFace Feature Extraction API is running 🚀",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "extract_image": "/extract/image",
            "extract_video": "/extract/video",
            "extract_batch": "/extract/batch"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    # Check if OpenFace binary exists
    openface_exists = os.path.exists(OPENFACE_BINARY)
    
    return {
        "status": "healthy" if openface_exists else "unhealthy",
        "openface_binary": OPENFACE_BINARY,
        "openface_available": openface_exists,
        "temp_dir": TEMP_DIR,
        "output_dir": OUTPUT_DIR
    }

@app.post("/extract/image")
async def extract_image(
    file: UploadFile = File(...),
    output_format: str = Form("csv"),
    session_id: Optional[str] = Form(None)
):
    """
    Extract facial features from a single image
    
    Args:
        file: Image file (JPEG, PNG)
        output_format: Output format (csv, json) - default: csv
        session_id: Optional session ID for organizing outputs
    
    Returns:
        JSON with extraction results and output file path
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temp directory for this request
    request_id = session_id or f"img_{os.urandom(8).hex()}"
    temp_request_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(temp_request_dir, exist_ok=True)
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_request_dir, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing image: {file.filename} for session {request_id}")
        
        # Run OpenFace FeatureExtraction
        output_subdir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(output_subdir, exist_ok=True)
        
        cmd = [
            OPENFACE_BINARY,
            "-f", input_path,
            "-out_dir", output_subdir,
            "-2Dfp",  # 2D facial landmarks
            "-3Dfp",  # 3D facial landmarks
            "-pdmparams",  # PDM parameters
            "-pose",  # Head pose
            "-aus",  # Action Units
            "-gaze",  # Gaze estimation
        ]
        
        # Execute OpenFace
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"OpenFace error: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenFace extraction failed: {result.stderr}"
            )
        
        # Find output CSV file
        csv_files = list(Path(output_subdir).glob("*.csv"))
        if not csv_files:
            raise HTTPException(
                status_code=500,
                detail="No output CSV generated"
            )
        
        output_csv = str(csv_files[0])
        logger.info(f"Extraction completed: {output_csv}")
        
        return {
            "status": "success",
            "message": "Feature extraction completed",
            "session_id": request_id,
            "input_file": file.filename,
            "output_file": output_csv,
            "output_format": "csv"
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"OpenFace timeout for {file.filename}")
        raise HTTPException(status_code=504, detail="Processing timeout")
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        if os.path.exists(temp_request_dir):
            shutil.rmtree(temp_request_dir, ignore_errors=True)

@app.post("/extract/video")
async def extract_video(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Extract facial features from a video file
    
    Args:
        file: Video file (MP4, AVI, MOV)
        session_id: Optional session ID for organizing outputs
    
    Returns:
        JSON with extraction results and output file path
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temp directory
    request_id = session_id or f"vid_{os.urandom(8).hex()}"
    temp_request_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(temp_request_dir, exist_ok=True)
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_request_dir, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing video: {file.filename} for session {request_id}")
        
        # Run OpenFace FeatureExtraction
        output_subdir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(output_subdir, exist_ok=True)
        
        cmd = [
            OPENFACE_BINARY,
            "-f", input_path,
            "-out_dir", output_subdir,
            "-2Dfp",
            "-3Dfp",
            "-pdmparams",
            "-pose",
            "-aus",
            "-gaze",
        ]
        
        # Execute OpenFace (longer timeout for videos)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=300  # 5 minutes
        )
        
        if result.returncode != 0:
            logger.error(f"OpenFace error: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenFace extraction failed: {result.stderr}"
            )
        
        # Find output CSV file
        csv_files = list(Path(output_subdir).glob("*.csv"))
        if not csv_files:
            raise HTTPException(
                status_code=500,
                detail="No output CSV generated"
            )
        
        output_csv = str(csv_files[0])
        logger.info(f"Video extraction completed: {output_csv}")
        
        return {
            "status": "success",
            "message": "Video feature extraction completed",
            "session_id": request_id,
            "input_file": file.filename,
            "output_file": output_csv,
            "output_format": "csv"
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"OpenFace timeout for {file.filename}")
        raise HTTPException(status_code=504, detail="Processing timeout (5min limit)")
    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        if os.path.exists(temp_request_dir):
            shutil.rmtree(temp_request_dir, ignore_errors=True)

@app.post("/extract/frames")
async def extract_frames(
    session_id: str = Form(...),
    frames_dir: str = Form(...)
):
    """
    Extract features from a directory of frame images
    
    Args:
        session_id: Session ID
        frames_dir: Path to directory containing frame images
    
    Returns:
        JSON with extraction results
    """
    if not os.path.exists(frames_dir):
        raise HTTPException(status_code=400, detail=f"Frames directory not found: {frames_dir}")
    
    # Check for images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(frames_dir).glob(ext))
    
    if not image_files:
        raise HTTPException(status_code=400, detail="No image files found in directory")
    
    logger.info(f"Processing {len(image_files)} frames for session {session_id}")
    
    try:
        # Run OpenFace on directory
        output_subdir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(output_subdir, exist_ok=True)
        
        cmd = [
            OPENFACE_BINARY,
            "-fdir", frames_dir,
            "-out_dir", output_subdir,
            "-2Dfp",
            "-3Dfp",
            "-pdmparams",
            "-pose",
            "-aus",
            "-gaze",
        ]
        
        # Execute OpenFace
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=600  # 10 minutes for batch
        )
        
        if result.returncode != 0:
            logger.error(f"OpenFace error: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenFace extraction failed: {result.stderr}"
            )
        
        # Find output CSV files
        csv_files = list(Path(output_subdir).glob("*.csv"))
        
        logger.info(f"Batch extraction completed: {len(csv_files)} CSV files generated")
        
        return {
            "status": "success",
            "message": "Batch feature extraction completed",
            "session_id": session_id,
            "frames_processed": len(image_files),
            "output_files": [str(f) for f in csv_files],
            "output_dir": output_subdir
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"OpenFace timeout for batch processing")
        raise HTTPException(status_code=504, detail="Processing timeout (10min limit)")
    except Exception as e:
        logger.error(f"Error processing frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{session_id}/{filename}")
def download_output(session_id: str, filename: str):
    """Download extracted feature file"""
    file_path = os.path.join(OUTPUT_DIR, session_id, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=filename
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting OpenFace API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
