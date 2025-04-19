import torch

from pathlib import Path
import logging

from .validation import compare_model_outputs
from ..configuration.configuration import Config
from .utils.minimal import export_minimal_model
from .utils.onnx import export_to_onnx
from .utils.torch_script import export_to_torch_script
from ..training.utils.logger import setup_logging

setup_logging(__name__)


def deploy_model(
        model: torch.nn.Module,
        model_name: str,
        device: torch.device,
        configuration: Config,
        measure_model: bool = False,
) -> None:
    destination_directory = Path(configuration.deployment.destination_directory)
    dummy_input = torch.randn(1, 3, configuration.deployment.resolution, configuration.deployment.resolution).to(device)

    # Apply structured pruning if specified in the configuration
    if configuration.deployment.optimizations.get("apply_pruning", False):
        from .optimization import apply_structured_pruning
        model = apply_structured_pruning(model)

    # Export the minimal model checkpoint
    export_minimal_model(
        model=model,
        directory=destination_directory,
        model_name=model_name,
        dummy_input=dummy_input,
        device=device,
    )

    export_to_torch_script(
        model=model,
        directory=destination_directory,
        model_name=model_name,
        dummy_input=dummy_input,
        device=device,
    )

    # Export the model to ONNX
    export_to_onnx(
        model=model,
        model_name=model_name,
        directory=destination_directory,
        input_shape=tuple(dummy_input.shape[1:]),
        device=device,
    )

    if measure_model:
        # Load exported TorchScript and ONNX model for comparison.
        ts_model = torch.jit.load(str(destination_directory / f"{model_name}_torch_script_{str(device)}.pt"))
        onnx_model_path = destination_directory / f"{model_name}_{str(device)}.onnx"
        compare_model_outputs(model, ts_model, onnx_model_path, dummy_input, device)

    logging.info("Production build completed successfully.")
