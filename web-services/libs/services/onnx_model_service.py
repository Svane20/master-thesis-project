from fastapi import UploadFile, HTTPException
import onnxruntime as ort
import time
import os
import asyncio
from typing import List
import numpy as np
import io
import zipfile
from PIL import Image

from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings
from libs.models.utils import get_model_checkpoint_path_based_on_device
from libs.replacement.replacement import do_sky_replacement
from libs.services.base import BaseModelService
from libs.logging import logger
from libs.metrics import MODEL_LOAD_TIME, SINGLE_INFERENCE_TIME, SINGLE_INFERENCE_TOTAL_TIME, BATCH_INFERENCE_TIME, \
    BATCH_INFERENCE_TOTAL_TIME, SKY_REPLACEMENT_INFERENCE_TIME, SKY_REPLACEMENT_TIME, SKY_REPLACEMENT_TOTAL_TIME, \
    BATCH_SKY_REPLACEMENT_INFERENCE_TIME, BATCH_SKY_REPLACEMENT_TIME, BATCH_SKY_REPLACEMENT_TOTAL_TIME
from libs.utils.processing import get_alpha_png_bytes, preprocess_image, preprocess_images, get_alphas_as_zip, \
    get_replacement_as_zip, get_image_png_bytes


def _postprocess_alphas(outputs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Post-process the alpha outputs from the model.

    Args:
        outputs (List[np.ndarray]): The model outputs.

    Returns:
        List[np.ndarray]: The post-processed alphas.
    """
    result = outputs[0]
    return [np.squeeze(result[i], axis=0) for i in range(result.shape[0])]


class OnnxModelService(BaseModelService):
    def __init__(self, settings: Settings, config: Configuration):
        """
        Initializes the OnnxModelService with the given configuration.

        Args:
            settings (Settings): The configuration settings for the service.
            config (Configuration): The model configuration.
        """
        super().__init__(settings=settings, config=config)

        self.session: ort.InferenceSession | None = None
        self.input_name = None
        self.output_name = None

    def load_model(self) -> None:
        """
        Load the ONNX model from disk and initialize runtime resources.
        """
        # Log the start time of model loading
        start = time.perf_counter()

        # Get the model path based on the device
        model_path = get_model_checkpoint_path_based_on_device(
            model_path=self.config.model.model_path,
            device=self.device,
        )

        # Load the ONNX model
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]

            self.session = ort.InferenceSession(model_path, opts, providers=providers)
        except Exception as e:
            logger.error({"event": "model_load_failed", "error": str(e)})
            raise e

        # The model is loaded successfully
        self.is_model_loaded = self.session is not None

        # Set the input and output names
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
            "providers": self.session.get_providers(),
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
        image_np = image_tensor.unsqueeze(0).numpy()

        # Run inference
        inf_start = time.perf_counter()
        outputs = await self._inference(image_np)

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

    async def batch_predict(self, files: List[UploadFile]) -> bytes:
        """
        Perform inference on multiple images and return the results as a ZIP archive.

        Args:
            files (List[UploadFile]): The list of files containing the images.

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
        batch_tensor = await preprocess_images(files, self.transforms)
        batch_np = batch_tensor.numpy()

        # Run inference
        inf_start = time.perf_counter()
        outputs = await self._inference(batch_np)

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
        image_np = image_tensor.unsqueeze(0).numpy()

        # Run inference
        inf_start = time.perf_counter()
        outputs = await self._inference(image_np)

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
        try:
            replaced_image, foreground = do_sky_replacement(image, alpha)
        except Exception as e:
            logger.error({"event": "sky_replacement_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail="Sky replacement failed")

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

    async def batch_sky_replacement(self, files: List[UploadFile], extra: bool = False) -> bytes:
        """
        Perform sky replacement on multiple images and return the results as a ZIP archive.

        Args:
            files (List[UploadFile]): The list of files containing the images.
            extra (bool): Whether to include extra information.

        Returns:
            bytes: The ZIP archive containing the results of the sky replacement.
        """
        if self.session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        if len(files) > self.settings.MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail="Batch size exceeds the maximum limit")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Read the files into memory
        cached = []
        for f in files:
            data = await f.read()
            cached.append((f.filename, data))
            f.file.seek(0)

        # Preprocess the images
        batch_tensor = await preprocess_images(files, self.transforms)
        batch_np = batch_tensor.numpy()

        # Run inference
        inf_start = time.perf_counter()
        outputs = await self._inference(batch_np)

        # Log the inference time
        inference_time = time.perf_counter() - inf_start
        BATCH_SKY_REPLACEMENT_INFERENCE_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(inference_time)
        logger.info({"event": "inference_completed", "time": inference_time})

        # Post-process the alphas
        alphas = _postprocess_alphas(outputs)

        # Sky replacement
        replacement_start = time.perf_counter()
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for (filename, image_bytes), alpha in zip(cached, alphas):
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                replaced, foreground = do_sky_replacement(img, alpha)

                base, _ = os.path.splitext(filename)
                zipf.writestr(f"{base}_replaced.png",
                              get_image_png_bytes(replaced))
                if extra:
                    zipf.writestr(f"{base}_alpha.png",
                                  get_alpha_png_bytes(alpha))
                    zipf.writestr(f"{base}_foreground.png",
                                  get_image_png_bytes(foreground))

        replacement_time = time.perf_counter() - replacement_start
        BATCH_SKY_REPLACEMENT_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(replacement_time)
        logger.info({"event": "replaced_image", "time": replacement_time})

        # Log the total time taken for the request
        total_time = time.perf_counter() - request_start
        BATCH_SKY_REPLACEMENT_TOTAL_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(total_time)

        # Return the zip archive bytes
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    async def _inference(self, input: np.ndarray) -> List[np.ndarray]:
        try:
            if self.use_gpu:
                return self.session.run(
                    [self.output_name],
                    {self.input_name: input}
                )
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    self.session.run,
                    [self.output_name],
                    {self.input_name: input},
                )
        except Exception as e:
            logger.error({"event": "inference_failed", "error": str(e)})
            raise HTTPException(status_code=500, detail="Inference failed")
