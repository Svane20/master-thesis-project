import torch.nn as nn

from pathlib import Path
import logging
from typing import Dict, Any
import platform

from libs.training.utils.checkpoint_utils import load_checkpoint

from unet_resnet.modeling.unet_resnet_34 import UNetResNet34


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
        import torch
        backend = "inductor" if platform.system() == "Linux" else "aot_eager"
        logging.info(f"Compiling the model with backend '{backend}'.")

        try:
            model = torch.compile(model, backend=backend)
            logging.info(f"Successfully compiled model with backend '{backend}'.")
        except Exception as e:
            logging.error(f"Model compilation with backend '{backend}' failed: {e}")

            # Suppress Dynamo errors and fall back to eager mode.
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True

            logging.info("Falling back to eager mode (without compilation).")

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

    return UNetResNet34(
        pretrained=image_encoder_config.get("pretrained", True),
        freeze_pretrained=image_encoder_config.get("freeze_pretrained", False),
    )
