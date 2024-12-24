from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf

from unet.configuration.dataset import DatasetConfig
from unet.configuration.model import ModelConfig
from unet.configuration.scratch import ScratchConfig
from unet.configuration.training.base import TrainConfig


@dataclass
class Config:
    scratch: Optional[ScratchConfig] = None
    dataset: Optional[DatasetConfig] = None
    training: Optional[TrainConfig] = None
    model: Optional[ModelConfig] = None


def load_configuration(configuration_path: Path) -> Config:
    """
    Load the configuration from the given path using OmegaConf.

    This function will work for both "training.yaml" (which includes
    scratch, training, optimizer) and "inference.yaml" (which might
    only include 'model') by using defaults and making fields optional.
    """
    if not configuration_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_path}")

    # Load the YAML configuration into an OmegaConf object
    cfg = OmegaConf.load(configuration_path)

    # Base structured config with defaults. Optional fields are None or empty.
    base = OmegaConf.structured(Config())

    # Merge the user config on top of the base
    merged = OmegaConf.merge(base, cfg)

    # Convert to a typed Config object
    return OmegaConf.to_object(merged)
