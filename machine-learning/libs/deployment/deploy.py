import torch

from pathlib import Path
import logging

from ..configuration.configuration import Config
from .utils.minimal import export_minimal_model
from .utils.onnx import export_to_onnx
from ..training.utils.logger import setup_logging

setup_logging(__name__)


def deploy_model(
        model: torch.nn.Module,
        model_name: str,
        device: torch.device,
        configuration: Config,
) -> None:
    destination_directory = Path(configuration.deployment.destination_directory)
    dummy_input = torch.randn(1, 3, configuration.deployment.resolution, configuration.deployment.resolution).to(device)

    # Export the minimal model checkpoint
    export_minimal_model(
        model=model,
        device=device,
        directory=destination_directory,
        model_name=model_name,
    )

    # Export the model to ONNX
    export_to_onnx(
        model=model,
        directory=destination_directory,
        model_name=model_name,
        dummy_input=dummy_input,
    )

    logging.info("Production build completed successfully.")
