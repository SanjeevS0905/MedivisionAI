 
import os
import io
import base64

import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# 1. APPLICATION
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


# --------------------------------------------------
# 2. MODEL LOADING
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best.onnx"
)

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
 # LEG FRACTURE MODEL
LEG_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "leg_fracture_model.onnx"
)

leg_session = ort.InferenceSession(
    LEG_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

leg_input_name = leg_session.get_inputs()[0].name

except Exception as error:
    raise RuntimeError(
        f"Could not load ONNX model: {error}"
    )


# --------------------------------------------------
# 3. MODEL SETTINGS
# --------------------------------------------------

IMAGE_SIZE = 640

CONFIDENCE_THRESHOLD = 0.10

# Updated from 0.45 to 0.35
IOU_THRESHOLD = 0.35

MAX_UPLOAD_SIZE = 10 * 1024 * 1024


# --------------------------------------------------
# 4. IOU CALCULATION
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
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area2 = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - intersection

    return intersection / (union + 1e-7)


# --------------------------------------------------
# 5. IMAGE PREPROCESSING
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
# 6. MODEL POSTPROCESSING
# --------------------------------------------------

 def postprocess(output, ratio, pad_x, pad_y, image_size,
                confidence_threshold= confidence_threshold,
                iou_threshold=iou_threshold ,
                "label": label):

    predictions = np.asarray(output)

    # Expected model output: (1, 5, 8400)
    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2:
        raise ValueError(
            "Unexpected model output dimensions."
        )

    if predictions.shape[0] == 5:
        predictions = predictions.T

    if predictions.shape[1] != 5:
        raise ValueError(
            "Unexpected model output format."
        )

    # Extract confidence scores
    scores = predictions[:, 4]

    valid = (
        np.isfinite(scores) &
        (scores >= CONFIDENCE_THRESHOLD)
    )

    predictions = predictions[valid]
    scores = scores[valid]

    if len(predictions) == 0:
        return []

    # Convert centre-based boxes to corner coordinates
    boxes = predictions[:, :4].copy()

    center_x = boxes[:, 0].copy()
    center_y = boxes[:, 1].copy()
    box_width = boxes[:, 2].copy()
    box_height = boxes[:, 3].copy()

    boxes[:, 0] = center_x - box_width / 2
    boxes[:, 1] = center_y - box_height / 2
    boxes[:, 2] = center_x + box_width / 2
    boxes[:, 3] = center_y + box_height / 2

    # Remove invalid boxes
    finite_boxes = np.all(
        np.isfinite(boxes),
        axis=1
    )

    boxes = boxes[finite_boxes]
    scores = scores[finite_boxes]

    if len(boxes) == 0:
        return []

    # --------------------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # --------------------------------------------------

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

        # Suppress boxes with IoU >= 0.35
        order = remaining[
            ious < IOU_THRESHOLD
        ]

    # --------------------------------------------------
    # CONVERT COORDINATES TO ORIGINAL IMAGE SIZE
    # --------------------------------------------------

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
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],
                "confidence": round(
                    float(scores[index]),
                    4
                ),
                "label": "Candidate region"
            })

    return results


# --------------------------------------------------
# 7. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "MediVision AI API is running",
        "model": "YOLOv8 ONNX",
        "supported_body_part": "hand"
    }


# --------------------------------------------------
# 8. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "MediVision AI backend is running",
        "model_loaded": session is not None
    }


# --------------------------------------------------
# 9. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    body_part: str = Form(...)
):

    # Check supported body part
    if body_part.strip().lower() != "hand":

        raise HTTPException(
            status_code=400,
            detail="Only hand X-rays are supported."
        )

    # Check declared content type
    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Upload a JPG or PNG image."
        )

    # Read at most 10 MB plus one byte
    image_bytes = await file.read(
        MAX_UPLOAD_SIZE + 1
    )

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit."
        )

    # Validate actual image format
    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as check_image:

            if check_image.format not in [
                "JPEG",
                "PNG"
            ]:

                raise HTTPException(
                    status_code=400,
                    detail="Only JPG and PNG images are supported."
                )

            check_image.verify()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image."
        )

    # Limit image dimensions
    if image.width > 10000 or image.height > 10000:

        raise HTTPException(
            status_code=400,
            detail="Image dimensions are too large."
        )

    # Preprocess and run ONNX inference
    try:

        tensor, ratio, pad_x, pad_y = preprocess(
            image
        )

        output = session.run(
            None,
            {
                input_name: tensor
            }
        )[0]

        predictions = postprocess(
            output,
            ratio,
            pad_x,
            pad_y,
            image.size
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Please try another image."
        )

    # --------------------------------------------------
    # 10. DRAW CANDIDATE REGIONS
    # --------------------------------------------------

    draw = ImageDraw.Draw(image)

    for prediction in predictions:

        draw.rectangle(
            prediction["box"],
            outline=(0, 255, 0),
            width=3
        )

    # --------------------------------------------------
    # 11. ENCODE ANNOTATED IMAGE
    # --------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # --------------------------------------------------
    # 12. API RESPONSE
    # --------------------------------------------------

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
        ],

        "total_regions": len(predictions),

        "message": (
            "Candidate regions detected."
            if predictions
            else "No candidate regions detected."
        ) 
    }

@app.post("/predict/leg")
async def predict_leg(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    tensor, ratio, pad_x, pad_y = preprocess(image)

    outputs = leg_session.run(
        None,
        {leg_input_name: tensor}
    )

    predictions = postprocess(
        outputs[0],
        ratio,
        pad_x,
        pad_y,
        image.size,
        confidence_threshold=0.25,
        iou_threshold=0.50,
        label="Candidate fracture region"
    )

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for item in predictions:
        box = item["box"]
        draw.rectangle(box, outline="green", width=3)

    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=90)

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
        ],
        "total_regions": len(predictions),
        "message": (
            "Candidate fracture regions detected."
            if predictions
            else "No candidate regions detected."
        )
    }
