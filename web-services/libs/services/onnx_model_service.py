from fastapi import UploadFile, HTTPException
import torch
import onnxruntime
import time
import os
import asyncio
from typing import List
import numpy as np

from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings
from libs.replacement.replacement import do_sky_replacement
from libs.services.base import BaseModelService
from libs.logging import logger
from libs.metrics import MODEL_LOAD_TIME, SINGLE_INFERENCE_TIME, SINGLE_INFERENCE_TOTAL_TIME, BATCH_INFERENCE_TIME, \
    BATCH_INFERENCE_TOTAL_TIME, SKY_REPLACEMENT_INFERENCE_TIME, SKY_REPLACEMENT_TIME, SKY_REPLACEMENT_TOTAL_TIME
from libs.utils.processing import get_alpha_png_bytes, preprocess_image, preprocess_images, get_alphas_as_zip, \
    get_replacement_as_zip, get_image_png_bytes
from libs.utils.transforms import get_transforms


def _postprocess_alphas(outputs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Post-process the alpha outputs from the model.

    Args:
        outputs (List[np.ndarray]): The model outputs.

    Returns:
        List[np.ndarray]: The post-processed alphas.
    """
    result = outputs[0]
    alphas = []
    for i in range(result.shape[0]):
        alpha = np.squeeze(result[i], axis=0)
        alphas.append(alpha)
    return alphas


class OnnxModelService(BaseModelService):
    def __init__(self, settings: Settings, config: Configuration):
        """
        Initializes the OnnxModelService with the given configuration.

        Args:
            settings (Settings): The configuration settings for the service.
        """
        super().__init__(settings=settings, config=config)

        self.session = None
        self.input_name = None
        self.output_name = None

        self.transforms = get_transforms(
            size=self.config.model.transforms.image_size,
            mean=self.config.model.transforms.mean,
            std=self.config.model.transforms.std,
        )
        self.use_gpu = torch.cuda.is_available() and self.settings.USE_GPU
        self.project_name = self.config.project_info.project_name
        self.model_type = self.config.project_info.model_type
        self.hardware = "GPU" if self.use_gpu else "CPU"

    def load_model(self) -> None:
        """
        Load the ONNX model from disk and initialize runtime resources.
        """
        start = time.perf_counter()

        try:
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = min(1, os.cpu_count() - 1)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(
                self.config.model.model_path,
                session_options,
                providers=providers
            )
        except Exception as e:
            logger.error({"event": "model_load_failed", "error": str(e)})
            raise e

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Log the model load time
        model_load_time = time.perf_counter() - start
        MODEL_LOAD_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(model_load_time)
        logger.info({
            "event": "model_loaded",
            "format": "ONNX",
            "providers": providers,
            "hardware": self.hardware,
            "model_load_time": model_load_time,
        })

    async def single_predict(self, file: UploadFile) -> bytes:
        """
        Perform inference on a single image and return the predicted alpha matte as png.

        Args:
            file (UploadFile): The file containing the image.

        Returns:
            bytes: The predicted alpha matte as a PNG image.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Load and preprocess the image
        image_tensor, _ = await preprocess_image(file, self.transforms)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(
                None,
                self.session.run,
                [self.output_name],
                {self.input_name: image_tensor.unsqueeze(0).numpy()},
            )
        except Exception as e:
            logger.error({"event": "inference_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail="Inference failed")

        # Log the inference time
        inference_time = time.perf_counter() - inf_start
        SINGLE_INFERENCE_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(inference_time)
        logger.info({"event": "inference_completed", "time": inference_time})

        # Post-process the alpha
        alpha = _postprocess_alphas(outputs)[0]

        # Log the total time taken for the request
        total_time = time.perf_counter() - request_start
        SINGLE_INFERENCE_TOTAL_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(total_time)
        logger.info({"event": "request_completed", "time": total_time})

        # Convert the alpha to PNG bytes
        return get_alpha_png_bytes(alpha)

    async def batch_predict(self, files: list[UploadFile]) -> bytes:
        """
        Perform inference on multiple images and return the results as a ZIP archive.

        Args:
            files (list[UploadFile]): The list of files containing the images.

        Returns:
            bytes: The ZIP archive containing the predicted alpha mattes.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if len(files) > self.settings.MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail="Batch size exceeds the maximum limit")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Preprocess the images
        batch = await preprocess_images(files, self.transforms)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(
                None,
                self.session.run,
                [self.output_name],
                {self.input_name: batch.numpy()},
            )
        except Exception as e:
            logger.error({"event": "inference_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail="Inference failed")

        # Log the inference time
        inference_time = time.perf_counter() - inf_start
        BATCH_INFERENCE_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(inference_time)
        logger.info({"event": "inference_completed", "time": inference_time})

        # Post-process the alphas
        alphas = _postprocess_alphas(outputs)

        # Log the total time taken for the request
        total_time = time.perf_counter() - request_start
        BATCH_INFERENCE_TOTAL_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(total_time)
        logger.info({"event": "request_completed", "time": total_time})

        # Convert the alphas to a ZIP archive
        return b"".join([chunk async for chunk in get_alphas_as_zip(alphas, files)])

    async def sky_replacement(self, file: UploadFile, extra: bool = False) -> bytes:
        """
        Perform sky replacement on the input image and return the result.

        Args:
            file (UploadFile): The file containing the image.
            extra (bool): Whether to include extra information.

        Returns:
            bytes: The result of the sky replacement.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Load and preprocess the image
        image_tensor, image = await preprocess_image(file, self.transforms)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(
                None,
                self.session.run,
                [self.output_name],
                {self.input_name: image_tensor.unsqueeze(0).numpy()},
            )
        except Exception as e:
            logger.error({"event": "inference_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail="Inference failed")

        # Log the inference time
        inference_time = time.perf_counter() - inf_start
        SKY_REPLACEMENT_INFERENCE_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(inference_time)
        logger.info({"event": "inference_completed", "time": inference_time})

        # Post-process the alpha
        alpha = _postprocess_alphas(outputs)[0]

        # Perform sky replacement
        replacement_start = time.perf_counter()
        replaced_image, foreground = do_sky_replacement(image, alpha)

        # Log the sky replacement time
        replacement_time = time.perf_counter() - replacement_start
        SKY_REPLACEMENT_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(replacement_time)
        logger.info({"event": "replaced_image", "time": replacement_time})

        # Convert the replaced image to PNG bytes
        png_bytes = get_image_png_bytes(replaced_image)

        # Log the total time taken for the request
        total_time = time.perf_counter() - request_start
        SKY_REPLACEMENT_TOTAL_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(total_time)
        logger.info({"event": "request_completed", "time": total_time})

        # If no extra information is requested, return the replaced image
        if not extra:
            return png_bytes

        # Convert the alphas to a ZIP archive
        return b"".join([chunk async for chunk in get_replacement_as_zip(png_bytes, alpha, foreground)])
