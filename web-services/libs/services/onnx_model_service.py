from fastapi import UploadFile, HTTPException
import onnxruntime
import time
import os
import numpy as np
import asyncio

from libs.fastapi.config import Settings
from libs.services.base import BaseModelService
from libs.logging import logger
from libs.metrics import MODEL_LOAD_TIME, INFERENCE_TIME, TOTAL_TIME
from libs.utils.processing import load_image_to_array, preprocess_image, postprocess_mask

# Image preprocessing parameters
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxModelService(BaseModelService):
    def __init__(self, config: Settings):
        """
        Initializes the OnnxModelService with the given configuration.

        Args:
            config (Settings): The configuration settings for the service.
        """
        super().__init__(config)

        self.session = None
        self.input_name = None
        self.output_name = None

    def load_model(self) -> None:
        """
        Load the ONNX model from disk and initialize runtime resources.
        """
        start = time.perf_counter()

        try:
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = min(1, os.cpu_count() - 1)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.config.USE_GPU else \
                ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(self.config.MODEL_PATH, session_options, providers=providers)
        except Exception as e:
            logger.error({"event": "model_load_failed", "error": str(e)})
            raise e

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        MODEL_LOAD_TIME.set(time.perf_counter() - start)
        logger.info({"event": "model_loaded", "format": "ONNX", "providers": providers})

    async def single_predict(self, file: UploadFile) -> bytes:
        request_start = time.perf_counter()

        try:
            img_array = load_image_to_array(file)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")

        orig_size = (img_array.shape[1], img_array.shape[0])

        # Preprocess the image
        input_tensor = preprocess_image(img_array, IMG_MEAN, IMG_STD)

        # Run inference
        loop = asyncio.get_running_loop()
        inf_start = time.perf_counter()
        outputs = await loop.run_in_executor(
            None,
            self.session.run,
            [self.output_name],
            {self.input_name: input_tensor}
        )
        INFERENCE_TIME.observe(time.perf_counter() - inf_start)

        # Post-process the mask
        mask_array = outputs[0]
        mask_bytes = postprocess_mask(mask_array)
        TOTAL_TIME.observe(time.perf_counter() - request_start)

        return mask_bytes
