import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
import io
from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.inference.predictor import InferencePredictor
from src.core.logger import logger

app = FastAPI(
    title="Solar Panel Defect Detection API",
    description="API for detecting defects in solar panels using YOLOv8",
    version="1.0.0"
)
# Initialize predictor lazily
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    logger.info("Initializing inference predictor...")
    try:
        predictor = InferencePredictor()
        logger.info("Predictor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Solar Panel Defect Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="File provided is not an image."
        )
    try:
        contents = await file.read()
        nparr = np.frombuffer(
            contents,
            np.uint8
        )
        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )
        results = predictor.predict_image(img)
        
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(
            f"Prediction failed: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
