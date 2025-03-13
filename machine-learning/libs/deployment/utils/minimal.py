import logging

import torch
import torch.nn as nn

from pathlib import Path

from ...utils.device import compile_model


def export_minimal_model(
        model: nn.Module,
        device: torch.device,
        model_name: str,
        directory: Path,
):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): Model.
        device (torch.device): Device.
        model_name (str): Model name.
        directory (Path): Directory to save the model checkpoint to.
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{model_name}_minimal.pt"

    # Set model to device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Save the model checkpoint
    with open(checkpoint_path, "wb") as f:
        torch.save(obj=model.state_dict(), f=f)
    logging.info(f"Minimal model checkpoint saved at {checkpoint_path}")


def export_model_with_compile(
        model: nn.Module,
        device: torch.device,
        model_name: str,
        directory: Path,
):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): Model.
        device (torch.device): Device.
        model_name (str): Model name.
        directory (Path): Directory to save the model checkpoint to.
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{model_name}_compiled.pt"

    # Set model to device
    model = model.to(device)

    # Compile the model
    model = compile_model(model)

    # Set the model to evaluation mode
    model.eval()

    # Save the model checkpoint
    with open(checkpoint_path, "wb") as f:
        torch.save(obj=model.state_dict(), f=f)
    logging.info(f"Compiled model checkpoint saved at {checkpoint_path}")
