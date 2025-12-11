import shutil
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from infer_adapter import PixelNeRFWrapper



app = FastAPI()

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


ai_engine = PixelNeRFWrapper(checkpoint_path=CHECKPOINT, conf_path=CONF_PATH)


@app.post("/reconstruct")
async def reconstruct(files: List[UploadFile] = File(...)):
    """
    Accept one or more input views. When multiple files are provided, they are
    all passed to render_video for multi-view encoding (matches PixelNeRFWrapper).
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    os.makedirs("uploads", exist_ok=True)
    file_paths: List[str] = []

    # Save uploads
    for file in files:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb+") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(file_path)

    try:
        # Generate GIF (change gif=False for MP4)
        output_path = ai_engine.render_video(file_paths if len(file_paths) > 1 else file_paths[0], gif=True)
        return FileResponse(output_path, media_type="image/gif", filename="spin.gif")
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


@app.get("/download/{filename}")
async def download_gif(filename: str):
    """
    Serve a generated GIF from the output directory.
    """
    safe_name = os.path.basename(filename)
    output_path = os.path.join("output", safe_name)

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(output_path, media_type="image/gif", filename=safe_name)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
