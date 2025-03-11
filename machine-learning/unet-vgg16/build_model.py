import torch
import torch.nn as nn

from pathlib import Path
import logging
from typing import Dict, Any

from libs.training.utils.checkpoint_utils import load_checkpoint
from modeling.image_encoder import ImageEncoder
from modeling.mask_decoder import MaskDecoder
from modeling.unet_vgg16_bn import UNetVGG16BN


def build_model_for_train(configuration: Dict[str, Any]) -> nn.Module:
    """
    Builds the model for training.

    Args:
        configuration (Dict[str, Any]): Configuration for the model.

    Returns:
        nn.Module: Model.
    """
    return _build(configuration)


def build_model(
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
        load_checkpoint(model, checkpoint_path, compile_model)

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
