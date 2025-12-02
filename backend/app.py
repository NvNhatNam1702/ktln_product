import shutil
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from infer_adapter import PixelNeRFWrapper

app = FastAPI()

# Configuration
CHECKPOINT = "pixel-nerf/checkpoints/sn64_unseen_a2" 
CONF_PATH ="/home/nampc/code/kltn/product/backend/pixel-nerf/conf/exp/sn64_unseen.conf"
# Initialize AI
ai_engine = PixelNeRFWrapper(checkpoint_path=CHECKPOINT, conf_path= CONF_PATH)

@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    # Save Upload
    with open(file_path, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Generate GIF
        # We assume the user wants a GIF. If MP4, change gif=False
        output_path = ai_engine.render_video(file_path, gif=True)
        
        return FileResponse(output_path, media_type="image/gif", filename="spin.gif")
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
