import torch
import torch.nn as nn
from typing import Dict, Any
import os

from libs.logging import logger


def build_model(
        configuration: Dict[str, Any],
        model_path: str,
        device: torch.device,
        is_torch_script: bool = False,
) -> nn.Module:
    if device.type not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid device: {device}")

    if not os.path.exists(model_path):
        raise ValueError(f"Checkpoint path {model_path} does not exist")

    # Load the model from the configuration
    if is_torch_script:
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = _get_model(configuration)
        _load_checkpoint(model, model_path)

    # Model the model to the device
    model.to(device)

    # Move the model to eval mode
    model.eval()

    return model


def _load_checkpoint(model: nn.Module, checkpoint_path: str):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Check if there are any errors when loading the state dictionary
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
    if missing_keys:
        logger.error(missing_keys)
        raise RuntimeError("Missing keys in checkpoint.")

    if unexpected_keys:
        logger.error(unexpected_keys)
        raise RuntimeError("Unexpected keys in checkpoint.")

    logger.info("Loaded checkpoint successfully")


def _get_model(configuration: Dict[str, Any]):
    if "model_name" not in configuration:
        raise ValueError("Model name is not provided in the configuration")

    model_name = configuration["model_name"]

    if model_name == "ResNet50Matte":
        from libs.models.resnet import ResNet50Matte
        return ResNet50Matte(configuration)
    elif model_name == "DPTSwinV2Tiny256Matte":
        from libs.models.dpt import DPTSwinV2Tiny256Matte
        return DPTSwinV2Tiny256Matte(configuration)
    elif model_name == "SwinMattingModel":
        from libs.models.swin import SwinMattingModel
        return SwinMattingModel(configuration)
    else:
        raise ValueError(f"Model {model_name} is not supported")
