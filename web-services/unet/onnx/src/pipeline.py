import torch
import torchvision.transforms as T
import onnxruntime as ort
from fastapi import UploadFile, HTTPException
from prometheus_client import Histogram

import io
import os
import asyncio
from PIL import Image
import numpy as np
from typing import List
import zipfile
import logging
import time

from src.config import get_configuration
from src.replacement.foreground_estimation import get_foreground_estimation
from src.replacement.replacement import replace_background

SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transforms = T.Compose([
    T.Resize(size=SIZE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# Global variables for the ONNX session and cached input/output names
session: ort.InferenceSession = None
input_name: str = None
output_name: str = None

# Histogram for tracking inference duration
model_startup_histogram = Histogram(
    "model_startup_latency_seconds",
    "Model startup latency in seconds",
    labelnames=["model"],
)

batch_inference_histogram = Histogram(
    "batch_inference_latency_seconds",
    "Batch inference latency in seconds",
    labelnames=["model"],
)

single_inference_histogram = Histogram(
    "single_inference_latency_seconds",
    "Single inference latency in seconds",
    labelnames=["model"],
)

sky_replacement_histogram = Histogram(
    "sky_replacement_latency_seconds",
    "Sky replacement (post-processing) latency in seconds",
    labelnames=["model"],
)

total_latency_histogram = Histogram(
    "total_latency_seconds",
    "Total processing latency in seconds (inference + sky replacement)",
    labelnames=["model"],
)
MODEL_PREFIX_NAME = "unet-onnx"


def load_model() -> None:
    """
    Load the ONNX model with dynamic provider selection (GPU if available).
    Cache input/output names.
    """
    global session, input_name, output_name

    # Load the configuration
    config = get_configuration()

    # Configure ONNX session options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = min(1, os.cpu_count() - 1)

    # Select providers based on GPU availability
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    try:
        with model_startup_histogram.labels(model=MODEL_PREFIX_NAME).time():
            session = ort.InferenceSession(config.MODEL_PATH, providers=providers)

        logging.info(f"ONNX model loaded with providers: {providers}")
    except Exception as e:
        logging.error("Failed to load ONNX model", exc_info=e)
        raise

    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logging.info(f"Cached input name: {input_name}, output name: {output_name}")
    except Exception as e:
        logging.error("Failed to cache input/output names", exc_info=e)
        raise


async def batch_predict(files: List[UploadFile]) -> bytes:
    """
    Asynchronously perform inference on a batch of images.
    Returns a ZIP archive (as bytes) containing the prediction masks.
    """
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Pre-process images concurrently
    batch = await _pre_process(files)

    # Offload blocking ONNX inference call to a thread
    loop = asyncio.get_running_loop()

    # Start timing before the inference call
    start_time = time.perf_counter()
    try:
        with batch_inference_histogram.labels(model=MODEL_PREFIX_NAME).time():
            outputs = await loop.run_in_executor(None, session.run, [output_name], {input_name: batch})
    except Exception as e:
        logging.error("ONNX inference failed", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # End timing after the inference call
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    logging.info(f"Batch inference time: {inference_time:.4f} seconds")

    masks = _post_process(outputs)

    # Stream ZIP archive creation using a generator to minimize memory usage
    async def zip_generator():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file, mask in zip(files, masks):
                base, _ = os.path.splitext(file.filename)
                filename = f"{base}_mask.png"
                png_bytes = _convert_mask_to_rgb(mask)
                zip_file.writestr(filename, png_bytes)
        buffer.seek(0)
        while True:
            chunk = buffer.read(4096)
            if not chunk:
                break
            yield chunk

    # Collect ZIP bytes
    zip_bytes = b"".join([chunk async for chunk in zip_generator()])

    return zip_bytes


async def single_predict(file: UploadFile) -> bytes:
    """
    Asynchronously perform inference on a single image.
    Returns the PNG image bytes of the predicted mask.
    """
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read and preprocess the single image
    image = await _read_file(file)
    try:
        tensor = transforms(image)
    except Exception as e:
        logging.error("Error processing image", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    # Add batch dimension: shape becomes [1, 3, 512, 512]
    batch_tensor = tensor.unsqueeze(0)
    batch_np = batch_tensor.numpy()

    loop = asyncio.get_running_loop()

    # Start timing before the inference call
    start_time = time.perf_counter()
    try:
        with single_inference_histogram.labels(model=MODEL_PREFIX_NAME).time():
            outputs = await loop.run_in_executor(None, session.run, [output_name], {input_name: batch_np})
    except Exception as e:
        logging.error("ONNX inference failed", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # End timing after the inference call
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    logging.info(f"Single inference time: {inference_time:.4f} seconds")

    # Post-process outputs; _post_process returns a list, take first mask
    masks = _post_process(outputs)
    mask = masks[0]
    png_bytes = _convert_mask_to_rgb(mask)

    return png_bytes


async def sky_replacement(file: UploadFile):
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read and preprocess the single image
    image = await _read_file(file)
    try:
        tensor = transforms(image)
    except Exception as e:
        logging.error("Error processing image", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    # Add batch dimension: shape becomes [1, 3, 512, 512]
    batch_tensor = tensor.unsqueeze(0)
    batch_np = batch_tensor.numpy()

    loop = asyncio.get_running_loop()

    with total_latency_histogram.labels(model=MODEL_PREFIX_NAME).time():
        total_start = time.perf_counter()

        # Inference timing using single_inference_histogram
        inference_start = time.perf_counter()
        try:
            with single_inference_histogram.labels(model=MODEL_PREFIX_NAME).time():
                outputs = await loop.run_in_executor(None, session.run, [output_name], {input_name: batch_np})
        except Exception as e:
            logging.error("ONNX inference failed", exc_info=e)
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start

        # Post-process outputs; _post_process returns a list, take first mask
        masks = _post_process(outputs)
        mask = masks[0]

        # Resize image and prepare for sky replacement
        image = image.resize(SIZE)
        image_array = np.array(image.resize(SIZE)) / 255.0

        # Sky replacement (post-processing) timing using its dedicated histogram
        replacement_start = time.perf_counter()
        with sky_replacement_histogram.labels(model=MODEL_PREFIX_NAME).time():
            foreground = get_foreground_estimation(image_array, mask)
            replaced = replace_background(foreground, mask)
        replacement_end = time.perf_counter()
        replacement_time = replacement_end - replacement_start

        total_end = time.perf_counter()
        total_time = total_end - total_start

    # Log the timings; these should match what the histograms record.
    logging.info(f"Inference time: {inference_time:.4f} seconds")
    logging.info(f"Sky replacement time: {replacement_time:.4f} seconds")
    logging.info(f"Total time: {total_time:.4f} seconds")

    # Convert both mask, foreground and replaced to PNG bytes.
    mask_png = _convert_mask_to_rgb(mask)
    foreground_png = _convert_image_to_png(foreground)
    replaced_png = _convert_image_to_png(replaced)

    # Create a ZIP archive containing both PNG files.
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("mask.png", mask_png)
        zip_file.writestr("foreground.png", foreground_png)
        zip_file.writestr("replaced.png", replaced_png)
    zip_buffer.seek(0)

    return zip_buffer.read()


def _post_process(outputs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Post-process the ONNX model outputs.
    Returns a list of 2D masks.
    """
    result = outputs[0]  # expected shape: (batch_size, 1, 512, 512)

    masks = []
    for i in range(result.shape[0]):
        logging.info(f"Mask {i} stats - min: {result[i].min()}, max: {result[i].max()}, mean: {result[i].mean()}")

        mask = np.squeeze(result[i], axis=0)  # shape: (512, 512)
        masks.append(mask)

    return masks


async def _pre_process(files: List[UploadFile]) -> np.ndarray:
    """
    Asynchronously pre-process input images for inference.
    Returns a NumPy array suitable for ONNX inference.
    """
    images = await asyncio.gather(*[_read_file(file) for file in files])

    batch_tensors = []
    for img in images:
        try:
            tensor = transforms(img)
            batch_tensors.append(tensor)
        except Exception as e:
            logging.error("Error processing image", exc_info=e)
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    # Stack tensors into batch shape: [N, 3, 512, 512]
    batch_tensor = torch.stack(batch_tensors)

    return batch_tensor.numpy()


def _convert_image_to_png(image: np.ndarray) -> bytes:
    """
    Convert a 3-channel image (or any properly shaped image) to PNG bytes.
    Assumes that the image is in [0, 1] range; if not, adjust accordingly.
    """
    # If image is float in [0,1], convert to uint8.
    image_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    pil_img = Image.fromarray(image_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def _convert_mask_to_rgb(mask: np.ndarray) -> bytes:
    """
    Convert a 2D grayscale mask to a 3-channel RGB PNG image with contrast stretching.
    """
    # Log mask statistics for debugging
    logging.info(f"Mask stats - min: {mask.min()}, max: {mask.max()}, mean: {mask.mean()}")

    # Contrast stretching: scale mask so that min becomes 0 and max becomes 1
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_uint8 = (norm_mask * 255).astype(np.uint8)
    rgb = np.stack([mask_uint8] * 3, axis=-1)
    pil_img = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


async def _read_file(file: UploadFile) -> Image.Image:
    """
    Asynchronously read a file and convert it to a PIL Image.
    """
    try:
        image_bytes = await file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logging.error(f"Error reading file {file.filename}: {e}", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
