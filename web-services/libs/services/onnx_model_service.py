from fastapi import UploadFile, HTTPException
import onnxruntime
import time
import os
import asyncio
from typing import List
import numpy as np

from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings
from libs.services.base import BaseModelService
from libs.logging import logger
from libs.metrics import MODEL_LOAD_TIME, INFERENCE_TIME, TOTAL_TIME
from libs.utils.processing import load_image, postprocess_mask
from libs.utils.transforms import get_transforms


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

    def load_model(self) -> None:
        """
        Load the ONNX model from disk and initialize runtime resources.
        """
        start = time.perf_counter()

        try:
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = min(1, os.cpu_count() - 1)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.settings.USE_GPU else \
                ["CPUExecutionProvider"]
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
        MODEL_LOAD_TIME.set(model_load_time)
        logger.info({
            "event": "model_loaded",
            "format": "ONNX",
            "providers": providers,
            "hardware": "GPU" if self.settings.USE_GPU else "CPU",
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
        # Log the start time of the request
        request_start = time.perf_counter()

        # Load and preprocess the image
        try:
            pil_image = await load_image(file)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")
        try:
            image_tensor = self.transforms(pil_image)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to preprocess image")

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        outputs = await loop.run_in_executor(
            None,
            self.session.run,
            [self.output_name],
            {self.input_name: image_tensor.unsqueeze(0).numpy()},
        )

        # Log the inference time
        inference_time = time.perf_counter() - inf_start
        INFERENCE_TIME.observe(inference_time)
        logger.info({"event": "inference_completed", "time": inference_time})

        # Post-process the mask
        masks = self._post_process_mask(outputs)[0]
        mask_bytes = postprocess_mask(masks)

        # Log the total time taken for the request
        total_time = time.perf_counter() - request_start
        TOTAL_TIME.observe(total_time)
        logger.info({"event": "request_completed", "time": total_time})

        return mask_bytes

    def _post_process_mask(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Post-process the mask outputs from the model.

        Args:
            outputs (List[np.ndarray]): The model outputs.

        Returns:
            List[np.ndarray]: The post-processed masks.
        """
        result = outputs[0]
        masks = []
        for i in range(result.shape[0]):
            mask = np.squeeze(result[i], axis=0)
            masks.append(mask)
        return masks
