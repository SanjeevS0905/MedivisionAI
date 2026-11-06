
import os
import io
import base64

import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="MediVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sanjeevs0905.github.io"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "MediVision AI API is running"}


@app.get("/health")
def health():
    return {"status": "MediVision AI backend is running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    body_part: str = Form(...)
):
    if body_part.lower() != "hand":
        raise HTTPException(
            status_code=400,
            detail="Only hand X-rays are supported."
        )

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG or PNG."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB."
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    results = model.predict(
        source=np.array(image),
        conf=0.10,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = results[0]
    draw = ImageDraw.Draw(image)
    predictions = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )
            confidence = float(box.conf[0])

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 255, 0),
                width=3
            )

            predictions.append({
                "confidence": round(confidence, 4),
                "label": "Candidate region"
            })

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "annotated_image": (
            "data:image/jpeg;base64," + encoded
        ),
        "predictions": predictions
    }
