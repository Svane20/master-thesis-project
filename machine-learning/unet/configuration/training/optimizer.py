from pydantic.dataclasses import dataclass


@dataclass
class AMPConfig:
    enabled: bool
    amp_dtype: str

    def asdict(self):
        return {
            "enabled": self.enabled,
            "amp_dtype": self.amp_dtype,
        }


@dataclass
class GradientClipConfig:
    enabled: bool
    max_norm: float
    norm_type: int

    def asdict(self):
        return {
            "enabled": self.enabled,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
        }


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    amp: AMPConfig
    gradient_clip: GradientClipConfig

    def asdict(self):
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "amp": self.amp.asdict(),
            "gradient_clip": self.gradient_clip.asdict(),
        }
