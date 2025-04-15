from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings
from libs.services.base import BaseModelService


def get_model_service(settings: Settings, configuration: Configuration) -> BaseModelService:
    """
    Returns the model service based on the model type specified in the project info.

    Args:
        settings (Settings): The settings object containing configuration.
        configuration (Configuration): The configuration related to the project.

    Returns:
        BaseModelService: An instance of the appropriate model service.
    """
    model_type = configuration.project_info.model_type

    if model_type == "onnx":
        from libs.services.onnx_model_service import OnnxModelService
        return OnnxModelService(settings, configuration)
    elif model_type == "torchscript" or model_type == "pytorch":
        from libs.services.pytorch_model_service import PytorchModelService
        return PytorchModelService(settings, configuration)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
