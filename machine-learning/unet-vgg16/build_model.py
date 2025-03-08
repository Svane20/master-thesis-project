import torch
import torch.nn as nn

from pathlib import Path
import logging
from typing import Dict, Any

from .modeling.image_encoder import ImageEncoder
from .modeling.mask_decoder import MaskDecoder
from .modeling.unet_vgg16_bn import UNetVGG16BN


def build_unet_model_for_train(configuration: Dict[str, Any]) -> nn.Module:
    """
    Builds the model for training.

    Args:
        configuration (Dict[str, Any]): Configuration for the model.

    Returns:
        nn.Module: Model.
    """
    return _build(configuration)


def build_unet_model(
        configuration: Dict[str, Any],
        checkpoint_path: Path = None,
        compile_model: bool = True,
        device: str = "cuda",
        mode: str = "eval"
) -> nn.Module:
    """
    Builds the model.

    Args:
        configuration (Dict[str, Any]): Configuration for the model.
        checkpoint_path (pathlib.Path): Path to the checkpoint.
        compile_model (bool): Compile the model. Default is True.
        device (str): Device to run the model. Default is "cuda".
        mode (str): Mode to run the model. Default is "eval".

    Returns:
        nn.Module: Model.
    """
    assert device in ["cuda", "cpu"], f"Invalid device: {device}"
    assert mode in ["train", "eval"], f"Invalid mode: {mode}"

    logging.info(f"Building UNet model in {mode} mode on {device}")

    model = _build(configuration)
    model = model.to(device)

    if compile_model:
        model = torch.compile(model, backend="aot_eager")

    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, compile_model)

    if mode == "eval":
        model.eval()

    return model


def _build(configuration: Dict[str, Any]) -> nn.Module:
    """
    Builds the model.

    Args:
        configuration (Dict[str, Any]): Configuration for the model.

    Returns:
        nn.Module: Model.
    """
    image_encoder_config = configuration.get("image_encoder", {})
    mask_decoder_config = configuration.get("mask_decoder", {})

    return UNetVGG16BN(
        image_encoder=ImageEncoder(
            pretrained=image_encoder_config.get("pretrained", False),
            freeze_pretrained=image_encoder_config.get("freeze_pretrained", False),
        ),
        mask_decoder=MaskDecoder(
            out_channels=mask_decoder_config.get("out_channels", 1),
            dropout=mask_decoder_config.get("dropout", 0.0),
        )
    )


def _load_checkpoint(model: nn.Module, checkpoint_path: Path, is_compiled: bool) -> None:
    """
    Load checkpoint for the model.

    Args:
        model (Module): Model to load the checkpoint.
        checkpoint_path (Path): Path to the checkpoint.
        is_compiled (bool): True if model is compiled.

    Exceptions:
        RuntimeError: If missing or unexpected keys in the checkpoint
    """
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError("Checkpoint not found.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = checkpoint["model"]

    # If model has not been compiled, adjust key names
    if not is_compiled:
        model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError("Missing keys in checkpoint.")

    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError("Unexpected keys in checkpoint.")

    logging.info("Loaded checkpoint successfully")
