from pathlib import Path
from typing import Tuple

from configuration.configuration import Config, load_configuration


def load_config(current_directory: Path, configuration_path: str) -> Tuple[Config, Path]:
    # Get configuration path
    configuration_path = current_directory / configuration_path

    # Check if the configuration file exists
    if not configuration_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_path}")

    # Load the configuration
    configuration = load_configuration(configuration_path)

    # Checkpoint path
    checkpoint_path = current_directory / "checkpoints/unet.pt"

    # Check if the checkpoint file exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return configuration, checkpoint_path
