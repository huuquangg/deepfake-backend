# import os
# import shutil
# import subprocess
# from fastapi import UploadFile


# class OpenFaceService:
#     def __init__(
#         self,
#         container_name: str = "openface",
#         base_dir: str = "/home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/extractor/open_face",
#         user: str = "huuquangdang"
#     ):
#         """
#         Service to run OpenFace FeatureExtraction inside Docker.

#         Args:
#             container_name (str): Name of the running OpenFace container.
#             base_dir (str): Base path of this module (where input/output live).
#             user (str): Username on the host to set file ownership.
#         """
#         self.container_name = container_name
#         self.input_dir = os.path.join(base_dir, "input")
#         self.output_dir = os.path.join(base_dir, "output")
#         self.user = user

#         # Path to FeatureExtraction inside the container
#         self.feature_extraction_bin = "/usr/local/bin/FeatureExtraction"

#         # Ensure directories exist
#         os.makedirs(self.input_dir, exist_ok=True)
#         os.makedirs(self.output_dir, exist_ok=True)

#     def _clear_dir(self, path: str):
#         """Remove all old files/folders in a directory."""
#         if os.path.exists(path):
#             subprocess.run(
#                 f"sudo rm -rf {path}/*",
#                 shell=True,
#                 check=True
#             )

#     def _run_feature_extraction(self, image_filename: str):
#         """Run FeatureExtraction on a file already inside mounted input_dir."""
#         container_image = f"/data/input/{image_filename}"
#         container_output = "/data/output"

#         # Clear old output
#         self._clear_dir(self.output_dir)

#         # Full command with all parameters enabled
#         cmd = [
#             "docker", "exec", "-i", self.container_name,
#             self.feature_extraction_bin,
#             "-f", container_image,
#             "-out_dir", container_output,
#             "-simalign", f"{container_output}/aligned",
#             "-gaze",        # enable gaze estimation
#             "-pose",        # enable head pose estimation
#             "-2Dfp",        # 2D facial landmarks
#             "-3Dfp",        # 3D facial landmarks
#             "-aus",         # Action Units (AUs)
#             # "-pdmparams",   # Shape/model parameters
#             # "-hogalign",    # HOG features for aligned faces
#             # "-tracked"      # Tracked bounding boxes
#         ]

#         # Run the command and capture output
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         if result.returncode != 0:
#             raise RuntimeError(
#                 f"OpenFace FeatureExtraction failed with exit code {result.returncode}\n"
#                 f"Stdout: {result.stdout}\nStderr: {result.stderr}"
#             )

#         # Fix permissions (make output owned by host user)
#         chown_cmd = ["sudo", "chown", "-R", f"{self.user}:{self.user}", self.output_dir]
#         subprocess.run(chown_cmd, check=True)

#     def analyze_image(self, file: UploadFile):
#         """
#         Analyze a single image from REST request.

#         New results will override the previous ones (both input and output).
#         """
#         # Clear input so only 1 file is stored
#         self._clear_dir(self.input_dir)

#         input_path = os.path.join(self.input_dir, file.filename)

#         # Save uploaded image to disk
#         with open(input_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         # Run OpenFace (overwrites output each time)
#         self._run_feature_extraction(file.filename)

#         return {
#             "status": "ok",
#             "input_file": file.filename,
#             "output_dir": self.output_dir,
#             "aligned_faces_dir": os.path.join(self.output_dir, "aligned")
#         }

