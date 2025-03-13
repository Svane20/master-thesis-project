from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from omegaconf import OmegaConf
from typing import Dict, Any
from enum import Enum
import platform

from .dataset import DatasetConfig
from .deployment.root import DeploymentConfig
from .evaluation.root import EvaluationConfig
from .scratch import ScratchConfig
from .training.root import TrainConfig


@dataclass
class Config:
    scratch: Optional[ScratchConfig] = None
    dataset: Optional[DatasetConfig] = None
    training: Optional[TrainConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    deployment: Optional[DeploymentConfig] = None
    model: Optional[Dict[str, Any]] = None

    def asdict(self):
        return {
            "scratch": self.scratch.asdict() if self.scratch is not None else None,
            "dataset": self.dataset.asdict() if self.dataset is not None else None,
            "training": self.training.asdict() if self.training is not None else None,
        }


class ConfigurationMode(str, Enum):
    Training = "training"
    Evaluation = "evaluation"
    Deployment = "deployment"


def get_configuration_and_checkpoint_path(mode: ConfigurationMode, suffix: str) -> Tuple[Config, Path]:
    config = get_configuration(mode=mode, suffix=suffix)

    if mode == ConfigurationMode.Training:
        checkpoint_path = Path(config.training.checkpoint.checkpoint_path)
    elif mode == ConfigurationMode.Evaluation:
        checkpoint_path = Path(config.evaluation.checkpoint_path)
    elif mode == ConfigurationMode.Deployment:
        checkpoint_path = Path(config.deployment.checkpoint_path)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return config, checkpoint_path


def get_configuration(mode: ConfigurationMode, suffix: str) -> Config:
    configuration_path = _get_base_configuration_path(mode=mode, suffix=suffix)

    return _load_configuration(configuration_path)


def _get_base_configuration_path(mode: ConfigurationMode, suffix: str) -> Path:
    base_directory = Path(__file__).resolve().parent.parent.parent
    os_suffix = "windows" if platform.system() == "Windows" else "linux"

    base_names = {
        ConfigurationMode.Training: "training",
        ConfigurationMode.Evaluation: "evaluation",
        ConfigurationMode.Deployment: "deployment",
    }

    try:
        base_name = base_names[mode]
    except KeyError:
        raise ValueError(f"Unknown mode: {mode}")

    return base_directory / suffix / "configs" / f"{base_name}_{os_suffix}.yaml"


def _load_configuration(configuration_path: Path) -> Config:
    """
    Load the configuration from the given path using OmegaConf.

    This function will work for both "training.yaml" (which includes
    scratch, training, optimizer) and "evaluation.yaml" (which might
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
