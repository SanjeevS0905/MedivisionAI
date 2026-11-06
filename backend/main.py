 
import os
import io
import base64

import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="MediVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sanjeevs0905.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.onnx"
)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"ONNX model not found: {MODEL_PATH}")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


def calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (
        boxes[:, 3] - boxes[:, 1]
    )

    return intersection / (
        area1 + area2 - intersection + 1e-7
    )


def preprocess(image):
    original_width, original_height = image.size
    size = 640

    ratio = min(
        size / original_width,
        size / original_height
    )

    new_width = round(original_width * ratio)
    new_height = round(original_height * ratio)

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new("RGB", (size, size), (114, 114, 114))

    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return array, ratio, pad_x, pad_y


def postprocess(output, ratio, pad_x, pad_y, image_size):
    predictions = output[0]

    if predictions.shape[0] == 5:
        predictions = predictions.T

    scores = predictions[:, 4]

    valid = scores >= 0.10
    predictions = predictions[valid]
    scores = scores[valid]

    if len(predictions) == 0:
        return []

    boxes = predictions[:, :4].copy()

    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + predictions[:, 2]
    boxes[:, 3] = boxes[:, 1] + predictions[:, 3]

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0 and len(keep) < 100:
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]
        ious = calculate_iou(
            boxes[current],
            boxes[remaining]
        )

        order = remaining[ious < 0.45]

    results = []
    width, height = image_size

    for index in keep:
        x1, y1, x2, y2 = boxes[index]

        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 > x1 and y2 > y1:
            results.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(scores[index]), 4),
                "label": "Candidate region"
            })

    return results


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

    tensor, ratio, pad_x, pad_y = preprocess(image)

    output = session.run(
        None,
        {input_name: tensor}
    )[0]

    predictions = postprocess(
        output,
        ratio,
        pad_x,
        pad_y,
        image.size
    )

    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        draw.rectangle(
            prediction["box"],
            outline=(0, 255, 0),
            width=3
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "annotated_image": "data:image/jpeg;base64," + encoded,
        "predictions": [
            {
                "confidence": item["confidence"],
                "label": item["label"]
            }
            for item in predictions
        ]
    }
