from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager

import onnxruntime as ort
from PIL import Image
import io
import numpy as np
from functools import lru_cache
import cv2
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("fastapi-app")

# Global model session
session: ort.InferenceSession


# Lifespan event handler
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global session

    try:
        session = ort.InferenceSession("unet.onnx")
        logger.info("ONNX model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise e

    yield


app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrumentator setup for Prometheus metrics
Instrumentator().instrument(app).expose(app)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess an image for inference.

    Args:
        image_bytes (bytes): Input image

    Returns:
        np.ndarray: Preprocessed image
    """
    # Decode and resize the image
    np_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    resized_image = cv2.resize(np_image, (224, 224))

    # Normalize and convert to tensor
    normalized_image = resized_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (normalized_image - mean) / std

    # Transpose to CHW format and add batch dimension
    return np.transpose(normalized_image, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


@lru_cache(maxsize=128)
def cached_preprocess(image_bytes: bytes) -> np.ndarray:
    return preprocess_image(image_bytes)


def postprocess_output(output: np.ndarray) -> np.ndarray:
    """
    Postprocess the model output.

    Args:
        output (np.ndarray): Model output

    Returns:
        np.ndarray: Post-processed binary mask
    """
    mask = output.squeeze()  # Remove batch and channel dimensions

    return (mask > 0.5).astype(np.uint8)


def remove_background(image_bytes: bytes, mask: np.ndarray) -> bytes:
    """
    Remove the background from the input image using the mask.

    Args:
        image_bytes (bytes): Original image bytes
        mask (np.ndarray): Binary mask

    Returns:
        bytes: Image bytes with transparent background
    """
    # Load the original image
    original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    # Verify image dimensions
    if original_image is None:
        raise ValueError("Failed to decode the image.")

    # Resize the mask to match the original image dimensions
    original_size = (original_image.shape[1], original_image.shape[0])
    resized_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Create the alpha channel
    alpha_channel = (resized_mask * 255).astype(np.uint8)

    # Add alpha channel if not present
    if original_image.shape[2] == 3:  # RGB
        transparent_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    else:
        transparent_image = original_image

    # Apply the alpha channel
    transparent_image[:, :, 3] = alpha_channel

    # Encode the image as PNG
    is_success, buffer = cv2.imencode(".png", transparent_image)
    if not is_success:
        raise ValueError("Failed to encode the transparent image.")
    return buffer.tobytes()

@app.get("/health", summary="Health Check", description="Check the health of the service")
async def health_check():
    return {"status": "ok"}


@app.post(
    "/api/v1/predict",
    summary="Predict Image Mask",
    description="Upload an image to predict its binary mask",
)
async def predict(file: UploadFile = File(...)):
    MAX_FILE_SIZE_MB = 5

    # Check file size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size is too large")

    try:
        # Preprocess image
        image_tensor = cached_preprocess(image_bytes)

        # Run inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: image_tensor})[0]

        # Postprocess output
        predicted_mask = postprocess_output(result)

        # Convert mask to image and return as PNG
        mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        byte_stream = io.BytesIO()
        mask_image.save(byte_stream, format="PNG")
        byte_stream.seek(0)

        return StreamingResponse(byte_stream, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post(
    "/api/v1/remove-background",
    summary="Remove Background",
    description="Upload an image to remove its background using the model",
)
async def remove_background_endpoint(file: UploadFile = File(...)):
    MAX_FILE_SIZE_MB = 5
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size is too large")

    try:
        # Preprocess image and run inference
        image_tensor = cached_preprocess(image_bytes)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: image_tensor})[0]
        predicted_mask = postprocess_output(result)

        # Remove background
        transparent_image_bytes = remove_background(image_bytes, predicted_mask)

        return StreamingResponse(io.BytesIO(transparent_image_bytes), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
