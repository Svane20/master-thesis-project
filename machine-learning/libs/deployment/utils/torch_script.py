import torch

from pathlib import Path
import logging

from libs.deployment.profiling import measure_latency, measure_memory_usage


def export_to_torch_script(
        model: torch.nn.Module,
        model_name: str,
        directory: Path,
        dummy_input: torch.Tensor,
        device: torch.device,
) -> None:
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}_{str(device)}_torch_script.pt"

    # Set model to evaluation mode
    model.eval()

    # Trace the model and freeze it to optimize for inference
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model = torch.jit.freeze(traced_model)

    # Save the model checkpoint
    with open(save_path, "wb") as f:
        torch.jit.save(m=traced_model, f=f)

    # Measure latency of the exported model
    measure_latency(traced_model, dummy_input)
    measure_memory_usage(dummy_input)

    logging.info(f"Torch scripted model checkpoint saved at {save_path}")
