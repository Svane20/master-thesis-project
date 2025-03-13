import torch
import torch.nn as nn

from pathlib import Path
import logging
from typing import Dict, Any

from libs.training.utils.checkpoint_utils import load_checkpoint
from libs.utils.device import compile_model

from unet.modeling.unet import UNet


def build_model_for_train(configuration: Dict[str, Any]) -> nn.Module:
    """
    Builds the model for training.

    Args:
        configuration (Dict[str, Any]): Configuration for the model.

    Returns:
        nn.Module: Model.
    """
    return _build(configuration)


def build_model_for_evaluation(
        configuration: Dict[str, Any],
        checkpoint_path: Path,
        device: torch.device
) -> nn.Module:
    return _build_model(
        configuration=configuration,
        checkpoint_path=checkpoint_path,
        should_compile=False,
        device=str(device),
        mode="eval"
    )


def build_model_for_deployment(
        configuration: Dict[str, Any],
        checkpoint_path: Path,
        device: torch.device
) -> nn.Module:
    return _build_model(
        configuration=configuration,
        checkpoint_path=checkpoint_path,
        should_compile=False,
        device=str(device),
        mode="eval"
    )


def _build_model(
        configuration: Dict[str, Any],
        checkpoint_path: Path = None,
        should_compile: bool = True,
        device: str = "cuda",
        mode: str = "eval"
) -> nn.Module:
    """
    Builds the model.

    Args:
        configuration (Dict[str, Any]): Configuration for the model.
        checkpoint_path (pathlib.Path): Path to the checkpoint.
        should_compile (bool): Compile the model. Default is True.
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

    if should_compile:
        model = compile_model(model)

    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, should_compile)

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
    return UNet(configuration)
