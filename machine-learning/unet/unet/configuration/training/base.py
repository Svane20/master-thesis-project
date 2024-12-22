from dataclasses import dataclass
from typing import Optional, List

from unet.configuration.training.criterion import CriterionConfig
from unet.configuration.training.optimizer import OptimizerConfig


@dataclass
class SchedulerConfig:
    name: str
    t_max: int
    eta_min: float


@dataclass
class WandbConfig:
    enabled: bool
    project: str
    entity: str
    tags: List[str]
    notes: str
    group: str
    job_type: str


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    verbose: bool
    patience: int
    min_delta: float
    monitor: str
    mode: str


@dataclass
class LoggingConfig:
    wandb: WandbConfig
    log_directory: str
    log_freq: int


@dataclass
class CheckpointConfig:
    save_directory: str
    checkpoint_path: str
    resume_from: Optional[str]


@dataclass
class TrainConfig:
    max_epochs: int
    warmup_epochs: int
    accelerator: str
    seed: int
    compile_model: bool

    criterion: CriterionConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
