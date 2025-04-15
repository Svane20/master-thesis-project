from fastapi import UploadFile, HTTPException
import torch
import asyncio
from typing import List
import time

from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings
from libs.metrics import MODEL_LOAD_TIME, SINGLE_INFERENCE_TIME, SINGLE_INFERENCE_TOTAL_TIME, BATCH_INFERENCE_TIME, \
    BATCH_INFERENCE_TOTAL_TIME, SKY_REPLACEMENT_INFERENCE_TIME, SKY_REPLACEMENT_TIME, SKY_REPLACEMENT_TOTAL_TIME
from libs.models.utils import build_model
from libs.replacement.replacement import do_sky_replacement
from libs.services.base import BaseModelService
from libs.logging import logger
from libs.utils.processing import preprocess_image, get_alpha_png_bytes, preprocess_images, get_alphas_as_zip, \
    get_image_png_bytes, get_replacement_as_zip


def _run_inference(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(image_tensor)


class PytorchModelService(BaseModelService):
    def __init__(self, settings: Settings, config: Configuration):
        """
        Initializes the TorchScriptModelService with the given configuration.

        Args:
            settings (Settings): The configuration settings for the service.
            config (Configuration): The model configuration.
        """
        super().__init__(settings=settings, config=config)

        self.model = None

        model_configuration = config.model.model_configuration
        if model_configuration is None:
            raise ValueError("Model configuration is missing.")

        self.model_configuration = model_configuration
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.is_torch_script = config.project_info.model_type == "torchscript"

    def load_model(self) -> None:
        """
        Load the pytorch model from disk and initialize runtime resources.
        """
        # Log the start time of model loading
        start = time.perf_counter()

        try:
            self.model = build_model(
                configuration=self.model_configuration,
                model_path=self.config.model.model_path,
                device=self.device,
                is_torch_script=self.is_torch_script,
            )
        except Exception as e:
            logger.error({"event": "model_load_failed", "error": str(e)})
            raise e

        # Log the model load time
        model_load_time = time.perf_counter() - start
        MODEL_LOAD_TIME.labels(
            model=self.project_name,
            type=self.model_type,
            hardware=self.hardware
        ).observe(model_load_time)

        current_format = "TorchScript" if self.is_torch_script else "PyTorch"
        logger.info({
            "event": "model_loaded",
            "format": current_format,
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
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Load and preprocess the image
        image_tensor, _ = await preprocess_image(file, self.transforms)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        try:
            output = await loop.run_in_executor(
                None, _run_inference, self.model, image_tensor
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
        alpha = output.detach().cpu().numpy()

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
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if len(files) > self.settings.MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail="Batch size exceeds the maximum limit")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Preprocess the images
        image_batch = await preprocess_images(files, self.transforms)
        image_batch = image_batch.to(self.device)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        try:
            output = await loop.run_in_executor(
                None, _run_inference, self.model, image_batch
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
        alphas = output.detach().cpu().numpy()

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
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Log the start time of the request
        request_start = time.perf_counter()

        # Load and preprocess the image
        image_tensor, image = await preprocess_image(file, self.transforms)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        try:
            output = await loop.run_in_executor(
                None, _run_inference, self.model, image_tensor
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
        alpha = output.detach().cpu().numpy().squeeze()

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
