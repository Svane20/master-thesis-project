import torch
import torch.nn as nn

from pathlib import Path
import logging

from ..profiling import measure_latency, measure_memory_usage


def export_minimal_model(
        model: nn.Module,
        model_name: str,
        directory: Path,
        dummy_input: torch.Tensor,
        device: torch.device,
):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): Model.
        model_name (str): Model name.
        directory (Path): Directory to save the model checkpoint to.
        dummy_input (torch.Tensor): Dummy input tensor.
        device (torch.device): Device to run the model on.
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{model_name}_{str(device)}_minimal.pt"

    # Set the model to evaluation mode
    model.eval()

    # Save the model checkpoint
    with open(checkpoint_path, "wb") as f:
        torch.save(obj=model.state_dict(), f=f)

    # Measure latency of the exported model
    if device.type == "cuda":
        measure_latency(model, dummy_input)
        measure_memory_usage(dummy_input)

    logging.info(f"Minimal model checkpoint saved at {checkpoint_path}")
