import os
import io
import time
import asyncio
import zipfile
import logging
import numpy as np
import torch
import torchvision.transforms as T
import onnxruntime as ort
from fastapi import UploadFile, HTTPException
from PIL import Image
from prometheus_client import Histogram

from src.config import get_configuration
from src.utils.image_utils import convert_image_to_png, convert_mask_to_rgb, read_image_async
from src.replacement.foreground_estimation import get_foreground_estimation
from src.replacement.replacement import replace_background

# Constants
SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Transformation pipeline
transforms = T.Compose([
    T.Resize(size=SIZE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

MODEL_PREFIX_NAME = "unet-onnx"

# Prometheus histograms
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


class ModelService:
    def __init__(self):
        self.model_path = None
        self.session = None
        self.input_name = None
        self.output_name = None

    def load_model(self) -> None:
        """
        Load the ONNX model with provider selection and cache input/output names.
        """
        # Load the configuration
        configuration = get_configuration()
        self.model_path = configuration.MODEL_PATH

        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = min(1, os.cpu_count() - 1)
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            with model_startup_histogram.labels(model=MODEL_PREFIX_NAME).time():
                self.session = ort.InferenceSession(self.model_path, providers=providers)
            logging.info(f"ONNX model loaded with providers: {providers}")
        except Exception as e:
            logging.error("Failed to load ONNX model", exc_info=e)
            raise e

        try:
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            logging.error("Failed to cache input/output names", exc_info=e)
            raise e

    async def batch_predict(self, files: list[UploadFile]) -> bytes:
        """
        Perform inference on a batch of images and return a ZIP archive of masks.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        batch = await self._pre_process(files)
        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()
        try:
            with batch_inference_histogram.labels(model=MODEL_PREFIX_NAME).time():
                outputs = await loop.run_in_executor(
                    None,
                    self.session.run,
                    [self.output_name],
                    {self.input_name: batch}
                )
        except Exception as e:
            logging.error("ONNX inference failed", exc_info=e)
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")
        end_time = time.perf_counter()
        logging.info(f"Batch inference time: {end_time - start_time:.4f} seconds")
        masks = self._post_process(outputs)

        async def zip_generator():
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file, mask in zip(files, masks):
                    base, _ = os.path.splitext(file.filename)
                    filename = f"{base}_mask.png"
                    png_bytes = convert_mask_to_rgb(mask)
                    zip_file.writestr(filename, png_bytes)
            buffer.seek(0)
            while True:
                chunk = buffer.read(4096)
                if not chunk:
                    break
                yield chunk

        zip_bytes = b"".join([chunk async for chunk in zip_generator()])
        return zip_bytes

    async def single_predict(self, file: UploadFile) -> bytes:
        """
        Perform inference on a single image and return the mask as a PNG.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        image = await read_image_async(file)
        try:
            tensor = transforms(image)
        except Exception as e:
            logging.error("Error processing image", exc_info=e)
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
        batch_tensor = tensor.unsqueeze(0)
        batch_np = batch_tensor.numpy()
        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()
        try:
            with single_inference_histogram.labels(model=MODEL_PREFIX_NAME).time():
                outputs = await loop.run_in_executor(
                    None,
                    self.session.run,
                    [self.output_name],
                    {self.input_name: batch_np}
                )
        except Exception as e:
            logging.error("ONNX inference failed", exc_info=e)
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")
        end_time = time.perf_counter()
        logging.info(f"Single inference time: {end_time - start_time:.4f} seconds")
        masks = self._post_process(outputs)
        mask = masks[0]
        png_bytes = convert_mask_to_rgb(mask)
        return png_bytes

    async def sky_replacement(self, file: UploadFile) -> bytes:
        """
        Perform sky replacement on an image and return a ZIP archive containing the mask, foreground, and replaced images.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        image = await read_image_async(file)
        try:
            tensor = transforms(image)
        except Exception as e:
            logging.error("Error processing image", exc_info=e)
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
        batch_tensor = tensor.unsqueeze(0)
        batch_np = batch_tensor.numpy()
        loop = asyncio.get_running_loop()

        with total_latency_histogram.labels(model=MODEL_PREFIX_NAME).time():
            total_start = time.perf_counter()
            inference_start = time.perf_counter()
            try:
                with single_inference_histogram.labels(model=MODEL_PREFIX_NAME).time():
                    outputs = await loop.run_in_executor(
                        None,
                        self.session.run,
                        [self.output_name],
                        {self.input_name: batch_np}
                    )
            except Exception as e:
                logging.error("ONNX inference failed", exc_info=e)
                raise HTTPException(status_code=500, detail=f"Inference error: {e}")
            inference_time = time.perf_counter() - inference_start

            masks = self._post_process(outputs)
            mask = masks[0]

            image = image.resize(SIZE)
            image_array = np.array(image) / 255.0

            replacement_start = time.perf_counter()
            with sky_replacement_histogram.labels(model=MODEL_PREFIX_NAME).time():
                foreground = get_foreground_estimation(image_array, mask)
                replaced = replace_background(foreground, mask)
            replacement_time = time.perf_counter() - replacement_start

            total_time = time.perf_counter() - total_start

        logging.info(f"Inference time: {inference_time:.4f} seconds")
        logging.info(f"Sky replacement time: {replacement_time:.4f} seconds")
        logging.info(f"Total time: {total_time:.4f} seconds")

        mask_png = convert_mask_to_rgb(mask)
        foreground_png = convert_image_to_png(foreground)
        replaced_png = convert_image_to_png(replaced)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("mask.png", mask_png)
            zip_file.writestr("foreground.png", foreground_png)
            zip_file.writestr("replaced.png", replaced_png)
        zip_buffer.seek(0)
        return zip_buffer.read()

    async def _pre_process(self, files: list[UploadFile]) -> np.ndarray:
        """
        Pre-process a list of image files into a NumPy array for inference.
        """
        images = await asyncio.gather(*[read_image_async(file) for file in files])
        batch_tensors = []
        for img in images:
            try:
                tensor = transforms(img)
                batch_tensors.append(tensor)
            except Exception as e:
                logging.error("Error processing image", exc_info=e)
                raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
        batch_tensor = torch.stack(batch_tensors)
        return batch_tensor.numpy()

    def _post_process(self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Post-process the ONNX model outputs to produce 2D masks.
        """
        result = outputs[0]
        masks = []
        for i in range(result.shape[0]):
            mask = np.squeeze(result[i], axis=0)
            masks.append(mask)
        return masks
