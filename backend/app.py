import shutil
import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from infer_adapter import PixelNeRFWrapper
from export_ply2 import export_ply


app = FastAPI()

# Configuration
CHECKPOINT = "pixel-nerf/checkpoints/sn64_unseen_a2"
CONF_PATH = "pixel-nerf/conf/exp/sn64_unseen.conf"


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


@app.post("/pointcloud")
async def pointcloud(
    transforms: UploadFile = File(..., description="transforms.json file"),
    images: List[UploadFile] = File(..., description="Multiple image files referenced in transforms.json"),
    download: bool = Form(True),
):
    """
    Upload the NeRF dataset (transforms.json + individual image files), generate a point cloud,
    and optionally download the resulting PLY file.
    """
    # Create a unique working directory for this request
    session_id = str(uuid.uuid4())
    work_dir = os.path.join("uploads", "pointcloud", session_id)
    os.makedirs(work_dir, exist_ok=True)

    # Save transforms.json
    transforms_path = os.path.join(work_dir, "transforms.json")
    try:
        with open(transforms_path, "wb+") as f:
            shutil.copyfileobj(transforms.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save transforms.json: {e}")

    # Save each uploaded image into the working directory
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded")

        for img in images:
            if not img.filename:
                continue
            img_path = os.path.join(work_dir, img.filename)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            with open(img_path, "wb+") as f:
                shutil.copyfileobj(img.file, f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded images: {e}")

    # The images_dir should be the root where transforms.json expects the image paths to start from.
    images_dir = work_dir

    # Output PLY path
    out_dir = os.path.join(work_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_ply = os.path.join(out_dir, "result.ply")

    try:
        export_ply(
            weights=CHECKPOINT + "/pixel_nerf_latest",
            transforms=transforms_path,
            out=out_ply,
            images_dir=images_dir,
            device="cuda:0",   # or "cpu"
            config=CONF_PATH,
            export_format="ply",
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if download:
        return FileResponse(out_ply, media_type="application/octet-stream", filename="result.ply")

    return {
        "message": "Point cloud generated",
        "file_path": out_ply,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
