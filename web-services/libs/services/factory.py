from libs.services.base import BaseModelService
from libs.fastapi.config import Settings


def get_model_service(config: Settings, model_type: str) -> BaseModelService:
    if model_type == "onnx":
        from libs.services.onnx_model_service import OnnxModelService
        return OnnxModelService(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
