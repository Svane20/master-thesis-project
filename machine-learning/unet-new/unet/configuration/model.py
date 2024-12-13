from dataclasses import dataclass


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
