from dataclasses import dataclass


@dataclass
class AMPConfig:
    enabled: bool
    amp_dtype: str


@dataclass
class GradientClipConfig:
    enabled: bool
    max_norm: float
    norm_type: int


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    amp: AMPConfig
    gradient_clip: GradientClipConfig
