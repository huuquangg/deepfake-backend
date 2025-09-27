import torch
import random
from pathlib import Path

# Load pretrained models once
BASE_PATH = Path(__file__).parent / "models"

# Dummy placeholders – replace with actual model loading
# cnn_model = torch.load(BASE_PATH / "cnn_model.pt", map_location="cpu")
# transformer_model = torch.load(BASE_PATH / "transformer_model.pt", map_location="cpu")

def run_cnn_inference(video_url: str):
    # TODO: implement feature extraction & inference
    # is_fake = random.choice([True, False])
    # confidence = random.uniform(0.6, 0.99)
    # return {"is_fake": is_fake, "confidence": confidence}
    return

def run_transformer_inference(video_url: str):
    # # TODO: implement feature extraction & inference
    # is_fake = random.choice([True, False])
    # confidence = random.uniform(0.7, 0.98)
    # return {"is_fake": is_fake, "confidence": confidence}
    return

def run_detection(video_url: str, method: str):
    # if method == "cnn":
    #     return run_cnn_inference(video_url)
    # elif method == "transformer":
    #     return run_transformer_inference(video_url)
    # else:
    #     raise ValueError(f"Unknown method: {method}")
    return