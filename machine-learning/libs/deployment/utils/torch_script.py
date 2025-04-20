import torch

from pathlib import Path
import logging

from ..profiling import measure_latency, measure_memory_usage


def export_to_torch_script(
        model: torch.nn.Module,
        model_name: str,
        directory: Path,
        dummy_input: torch.Tensor,
        device: torch.device,
        measure_model: bool = False,
) -> None:
    """
    Export the model to TorchScript format.

    Args:
        model (torch.nn.Module): Model to export.
        model_name (str): Name of the model.
        directory (Path): Directory to save the TorchScript model to.
        dummy_input (torch.Tensor): Dummy input tensor.
        device (torch.device): Device to run the model on.
        measure_model (bool): Whether to measure the model's latency and memory usage.
    """
    logging.info(f"Exporting torch scripted model...")

    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}_torch_script_{str(device)}.pt"

    # Set model to evaluation mode
    model.eval()

    # Trace the model and freeze it to optimize for inference
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model = torch.jit.freeze(traced_model)

    if device.type == "cuda":
        traced_model = torch.jit.optimize_for_inference(traced_model)

    # Save the model checkpoint
    with open(save_path, "wb") as f:
        torch.jit.save(m=traced_model, f=f)

    logging.info(f"Torch scripted model checkpoint saved at {save_path}")

    # Measure latency of the exported model
    if measure_model and device.type == "cuda":
        measure_latency(model, dummy_input)
        measure_memory_usage(dummy_input)

    logging.info(f"Finished exporting torch scripted model.")
