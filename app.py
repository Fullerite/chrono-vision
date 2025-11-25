import base64
import os
import uuid

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.pipeline import ChronoVisionPipeline

app = FastAPI(title="ChronoVision API")

pipeline = None

TEMP_DIR = os.getenv("TEMP_DIR", "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


def encode_image_to_base64(image_array):
    """Converts a numpy image (RGB) to base64 string"""
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


@app.on_event("startup")
def load_model():
    global pipeline
    COLORIZER_WEIGHTS = "models/main_efficientnet-b2_best.pt"
    GFPGAN_WEIGHTS = "models/GFPGANv1.3.pth"
    ESRGAN_WEIGHTS = "models/RealESRGAN_x4plus.pth"

    print("Loading models... this may take a moment.")
    pipeline = ChronoVisionPipeline(COLORIZER_WEIGHTS, GFPGAN_WEIGHTS, ESRGAN_WEIGHTS)
    print("Models loaded successfully.")


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(TEMP_DIR, unique_filename)

    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        original, colorized, restored = pipeline.run(temp_path)

        return {
            "original": encode_image_to_base64(original),
            "colorized": encode_image_to_base64(colorized),
            "restored": encode_image_to_base64(restored)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": "available" if pipeline else "loading"}
