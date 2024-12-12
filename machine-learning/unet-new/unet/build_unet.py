import torch
import torch.nn as nn

from pathlib import Path
import logging
from typing import Literal

from unet.configuration.model_configuration import Config
from unet.modeling.image_encoder import ImageEncoder
from unet.modeling.mask_decoder import MaskDecoder
from unet.modeling.unet import UNet


def build_model(
        configuration: Config,
        checkpoint_path: Path = None,
        compile_model: bool = True,
        device: Literal["cuda", "cpu"] = "cuda",
        mode: Literal["train", "train_only", "eval"] = "eval"
) -> nn.Module:
    """
    Build the UNet model.

    Args:
        configuration (Config): Configuration for the model.
        checkpoint_path (pathlib.Path): Path to the checkpoint.
        compile_model (bool): Compile the model. Default is True.
        device (str): Device to run the model. Default is "cuda".
        mode (str): Mode to run the model. Default is "eval".

    Returns:
        nn.Module: UNet model.
    """
    assert device in ["cuda", "cpu"], f"Invalid device: {device}"
    assert mode in ["train", "train_only", "eval"], f"Invalid mode: {mode}"

    logging.info(f"Building UNet model in {mode} mode on {device}")

    model = UNet(
        image_encoder=ImageEncoder(
            pretrained=configuration.model.image_encoder.pretrained
        ),
        mask_decoder=MaskDecoder(
            out_channels=configuration.model.mask_decoder.out_channels,
            dropout=configuration.model.mask_decoder.dropout
        )
    )

    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    model = model.to(device)

    if compile_model:
        try:
            model = torch.compile(model, backend="aot_eager")
        except AttributeError:
            logging.warning("torch.compile is not available; running without compilation.")

    if mode == "eval":
        model.eval()
    else:
        model.train()

    return model


def _load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """
    Load checkpoint for the model.

    Args:
        model (UNet): Model to load the checkpoint.
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
