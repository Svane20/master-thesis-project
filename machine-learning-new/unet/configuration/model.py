from pydantic.dataclasses import dataclass


@dataclass
class ImageEncoderConfig:
    pretrained: bool
    freeze_pretrained: bool


@dataclass
class MaskDecoderConfig:
    out_channels: int
    dropout: float


@dataclass
class ModelConfig:
    name: str
    image_encoder: ImageEncoderConfig
    mask_decoder: MaskDecoderConfig