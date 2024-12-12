from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ImageEncoderConfig:
    pretrained: bool


@dataclass
class MaskDecoderConfig:
    out_channels: int
    dropout: float


@dataclass
class ModelConfig:
    image_encoder: ImageEncoderConfig
    mask_decoder: MaskDecoderConfig


@dataclass
class Config:
    model: ModelConfig


def load_configuration(configuration_path: Path):
    """
    Load the configuration from the given path.

    Args:
        configuration_path (pathlib.Path): Path to the configuration.
    """
    if not configuration_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_path}")

    with open(configuration_path, "r") as file:
        data = yaml.safe_load(file)

        return Config(
            model=ModelConfig(
                image_encoder=ImageEncoderConfig(**data["model"]["image_encoder"]),
                mask_decoder=MaskDecoderConfig(**data["model"]["mask_decoder"]),
            ),
        )
