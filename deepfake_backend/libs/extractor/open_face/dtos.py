from pydantic import BaseModel

class OpenFaceResponse(BaseModel):
    status: bool
    input_file: str
    output_dir: str
    aligned_faces_dir: str