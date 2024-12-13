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


@dataclass
class SchedulerConfig:
    t_max: int
    eta_min: float


@dataclass
class WandbConfig:
    enabled: bool
    project: str


@dataclass
class LoggingConfig:
    wandb: WandbConfig
    log: str
    log_freq: int


@dataclass
class CheckpointConfig:
    save_directory: str


@dataclass
class TrainConfig:
    num_epochs: int
    accelerator: str
    seed: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
