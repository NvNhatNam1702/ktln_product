import shutil
import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from infer_adapter import PixelNeRFWrapper



app = FastAPI()

# Enable CORS to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins like ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT = "pixel-nerf/checkpoints/nam_final_2"
CONF_PATH = "pixel-nerf/conf/exp/multi_obj.conf"


# Initialize AI
ai_engine = PixelNeRFWrapper(checkpoint_path=CHECKPOINT, conf_path=CONF_PATH)


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    # Save upload
    with open(file_path, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Generate GIF (change gif=False for MP4)
        output_path = ai_engine.render_video(file_path, gif=True)
        return FileResponse(output_path, media_type="image/gif", filename="spin.gif")
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


@app.post("/extract_mesh")
async def extract_mesh(
    file: UploadFile = File(...),
    output_format: str = Form("obj"),
    resolution: int = Form(128),
    isosurface: float = Form(50.0),
):
    """
    Extract a 3D mesh from a single input image.
    
    Args:
        file: Input image file
        output_format: "obj" or "ply" (default: "obj")
        resolution: Grid resolution for marching cubes (default: 128)
        isosurface: Isosurface threshold for marching cubes (default: 50.0)
    
    Returns:
        Mesh file (OBJ or PLY format)
    """
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    # Save upload
    with open(file_path, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Extract mesh
        mesh_path = ai_engine.extract_mesh(
            file_path,
            output_format=output_format,
            resolution=resolution,
            isosurface=isosurface
        )
        
        # Determine media type
        if output_format.lower() == "ply":
            media_type = "application/octet-stream"
            filename = "mesh.ply"
        else:
            media_type = "application/octet-stream"
            filename = "mesh.obj"
        
        return FileResponse(mesh_path, media_type=media_type, filename=filename)
    except Exception as e:
        print(f"Error extracting mesh: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
