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
        measure_model: bool = False,
):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): Model.
        model_name (str): Model name.
        directory (Path): Directory to save the model checkpoint to.
        dummy_input (torch.Tensor): Dummy input tensor.
        device (torch.device): Device to run the model on.
        measure_model (bool): Whether to measure the model's latency and memory usage.
    """
    logging.info(f"Exporting minimal model...")

    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{model_name}_minimal_{str(device)}.pt"

    # Set the model to evaluation mode
    model.eval()

    # Save the model checkpoint
    with open(checkpoint_path, "wb") as f:
        torch.save(obj=model.state_dict(), f=f)

    logging.info(f"Minimal model checkpoint saved at {checkpoint_path}")

    # Measure latency of the exported model
    if measure_model and device.type == "cuda":
        measure_latency(model, dummy_input)
        measure_memory_usage(dummy_input)

    logging.info(f"Finished exporting minimal model.")
