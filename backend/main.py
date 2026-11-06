 
import io
import os
import base64

import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(title="MediVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sanjeevs0905.github.io"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best.onnx"
)

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# --------------------------------------------------
# 2. LOAD ONNX MODEL
# --------------------------------------------------

if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(
        f"ONNX model not found: {MODEL_PATH}"
    )

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print("MediVision ONNX model loaded successfully.")

except Exception as error:
    raise RuntimeError(
        f"Could not load ONNX model: {error}"
    )


# --------------------------------------------------
# 3. IOU CALCULATION
# --------------------------------------------------

def calculate_iou(box, boxes):

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area1 = (
        max(0, box[2] - box[0]) *
        max(0, box[3] - box[1])
    )

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - intersection

    return intersection / (union + 1e-7)


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess(image):

    original_width, original_height = image.size

    ratio = min(
        IMAGE_SIZE / original_width,
        IMAGE_SIZE / original_height
    )

    new_width = round(original_width * ratio)
    new_height = round(original_height * ratio)

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (114, 114, 114)
    )

    pad_x = (IMAGE_SIZE - new_width) // 2
    pad_y = (IMAGE_SIZE - new_height) // 2

    canvas.paste(
        resized,
        (pad_x, pad_y)
    )

    array = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    array = np.transpose(
        array,
        (2, 0, 1)
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array, ratio, pad_x, pad_y


# --------------------------------------------------
# 5. YOLO POSTPROCESSING
# --------------------------------------------------

def postprocess(
    output,
    ratio,
    pad_x,
    pad_y,
    image_size
):

    predictions = np.asarray(output)

    # Expected model output:
    # (1, 5, 8400) or (5, 8400)

    if predictions.ndim == 3:
        if predictions.shape[0] != 1:
            raise ValueError(
                "Unexpected ONNX batch dimension."
            )

        predictions = predictions[0]

    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected ONNX output shape: {predictions.shape}"
        )

    if predictions.shape[0] == 5:
        predictions = predictions.T

    elif predictions.shape[1] != 5:
        raise ValueError(
            f"Expected 5 output channels, got {predictions.shape}"
        )

    predictions = predictions.astype(
        np.float32,
        copy=False
    )

    # Remove invalid numerical predictions.
    valid_numbers = np.all(
        np.isfinite(predictions),
        axis=1
    )

    predictions = predictions[valid_numbers]

    if len(predictions) == 0:
        return []

    scores = predictions[:, 4]

    valid = scores >= CONFIDENCE_THRESHOLD

    predictions = predictions[valid]
    scores = scores[valid]

    if len(predictions) == 0:
        return []

    # Convert YOLO centre-format boxes:
    # cx, cy, width, height
    # into x1, y1, x2, y2.

    boxes = predictions[:, :4].copy()

    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2

    # Non-Maximum Suppression

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

        order = remaining[
            ious < IOU_THRESHOLD
        ]

    # Convert boxes back to original image coordinates.

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
                "box": [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2))
                ],
                "confidence": round(
                    float(scores[index]), 4
                ),
                "label": "Candidate region"
            })

    return results


# --------------------------------------------------
# 6. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "MediVision AI API is running"
    }


# --------------------------------------------------
# 7. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "MediVision AI backend is running",
        "model": "ONNX",
        "model_loaded": session is not None
    }


# --------------------------------------------------
# 8. X-RAY PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    body_part: str = Form(...)
):

    # Check selected body part.

    if body_part.lower() != "hand":

        raise HTTPException(
            status_code=400,
            detail="Only hand X-rays are supported."
        )

    # Check declared file type.

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    # Read at most 10 MB plus one byte.

    image_bytes = await file.read(
        MAX_FILE_SIZE + 1
    )

    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit."
        )

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # Verify that the file is a valid image.

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as uploaded_image:

            if uploaded_image.format not in [
                "JPEG",
                "PNG"
            ]:

                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG and PNG images are supported."
                )

            uploaded_image.verify()

        with Image.open(
            io.BytesIO(image_bytes)
        ) as uploaded_image:

            image = uploaded_image.convert("RGB")

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image."
        )

    # Preprocess image.

    try:

        tensor, ratio, pad_x, pad_y = preprocess(
            image
        )

        # Run ONNX inference.

        output = session.run(
            None,
            {
                input_name: tensor
            }
        )[0]

        # Decode predictions.

        predictions = postprocess(
            output,
            ratio,
            pad_x,
            pad_y,
            image.size
        )

    except Exception as error:

        print("Prediction error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs."
        )

    # Draw candidate regions.

    annotated_image = image.copy()

    draw = ImageDraw.Draw(
        annotated_image
    )

    for prediction in predictions:

        x1, y1, x2, y2 = prediction["box"]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0),
            width=3
        )

    # Encode annotated image.

    buffer = io.BytesIO()

    annotated_image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # Return website-compatible response.

    return {
        "annotated_image": (
            "data:image/jpeg;base64," + encoded
        ),

        "predictions": [
            {
                "confidence": item["confidence"],
                "label": item["label"]
            }
            for item in predictions
        ]
    }
