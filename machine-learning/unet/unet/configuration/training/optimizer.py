from dataclasses import dataclass

@dataclass
class AMPConfig:
    enabled: bool
    amp_dtype: str


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    amp: AMPConfig