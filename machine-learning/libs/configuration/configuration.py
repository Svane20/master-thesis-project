from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from omegaconf import OmegaConf
from typing import Dict, Any

from .dataset import DatasetConfig
from .deployment.root import DeploymentConfig
from .inference.root import InferenceConfig
from .scratch import ScratchConfig
from .training.root import TrainConfig


@dataclass
class Config:
    scratch: Optional[ScratchConfig] = None
    dataset: Optional[DatasetConfig] = None
    training: Optional[TrainConfig] = None
    inference: Optional[InferenceConfig] = None
    deployment: Optional[DeploymentConfig] = None
    model: Optional[Dict[str, Any]] = None

    def asdict(self):
        return {
            "scratch": self.scratch.asdict() if self.scratch is not None else None,
            "dataset": self.dataset.asdict() if self.dataset is not None else None,
            "training": self.training.asdict() if self.training is not None else None,
        }


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


def load_configuration_and_checkpoint(configuration_path: Path, is_deployment: bool = False) -> Tuple[Config, Path]:
    configuration = load_configuration(configuration_path)

    if is_deployment:
        checkpoint_path = Path(configuration.deployment.checkpoint_path)
    else:
        checkpoint_path = Path(configuration.inference.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return configuration, checkpoint_path
