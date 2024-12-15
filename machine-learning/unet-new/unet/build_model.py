import torch
import torch.nn as nn

from pathlib import Path
import logging

from unet.configuration.configuration import ModelConfig
from unet.modeling.image_encoder import ImageEncoder
from unet.modeling.mask_decoder import MaskDecoder
from unet.modeling.unet import UNet


def build_model_for_train(configuration: ModelConfig) -> nn.Module:
    """
    Builds the model for training.

    Args:
        configuration (Config): Configuration for the model.

    Returns:
        nn.Module: Model.
    """
    return _build(configuration)


def build_model(
        configuration: ModelConfig,
        checkpoint_path: Path = None,
        compile_model: bool = True,
        device: str = "cuda",
        mode: str = "eval"
) -> nn.Module:
    """
    Builds the model.

    Args:
        configuration (Config): Configuration for the model.
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
        _load_checkpoint(model, checkpoint_path)

    if mode == "eval":
        model.eval()

    return model


def _build(configuration: ModelConfig) -> nn.Module:
    """
    Builds the model.

    Args:
        configuration (ModelConfig): Configuration for the model.

    Returns:
        nn.Module: Model.
    """
    return UNet(
        image_encoder=ImageEncoder(
            pretrained=configuration.image_encoder.pretrained
        ),
        mask_decoder=MaskDecoder(
            out_channels=configuration.mask_decoder.out_channels,
            dropout=configuration.mask_decoder.dropout
        )
    )


def _load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """
    Load checkpoint for the model.

    Args:
        model (Module): Model to load the checkpoint.
        checkpoint_path (Path): Path to the checkpoint.

    Exceptions:
        RuntimeError: If missing or unexpected keys in the checkpoint
    """
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError("Checkpoint not found.")

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)

    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError("Missing keys in checkpoint.")

    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError("Unexpected keys in checkpoint.")

    logging.info("Loaded checkpoint successfully")
