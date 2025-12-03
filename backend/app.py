import shutil
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from infer_adapter import PixelNeRFWrapper
from export_ply2 import export_ply


app = FastAPI()

# Configuration
CHECKPOINT = "pixel-nerf/checkpoints/sn64_unseen_a2"
CONF_PATH = "/home/nampc/code/kltn/product/backend/pixel-nerf/conf/exp/sn64_unseen.conf"


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
async def pointcloud():
    """
    Example endpoint that calls export_ply().

    TODO: update `transforms_path` and `images_dir` to match your dataset layout.
    """
    # Example paths – adjust to your actual data
    transforms_path = "path/to/transforms.json"
    images_dir = "path/to/images_dir"
    out_ply = "output/result.ply"

    os.makedirs(os.path.dirname(out_ply), exist_ok=True)

    try:
        export_ply(
            weights=CHECKPOINT,
            transforms=transforms_path,
            out=out_ply,
            images_dir=images_dir,
            device="cuda:0",   # or "cpu"
            config=CONF_PATH,
            export_format="ply",
        )
        return FileResponse(out_ply, media_type="application/octet-stream", filename="result.ply")
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
