from pydantic.dataclasses import dataclass
from typing import Optional, List

from configuration.training.criterion import CriterionConfig
from configuration.training.optimizer import OptimizerConfig
from configuration.training.scheduler import SchedulerConfig


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

    def asdict(self):
        return {
            "enabled": self.enabled,
            "verbose": self.verbose,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "mode": self.mode
        }


@dataclass
class LoggingConfig:
    wandb: WandbConfig
    log_directory: str
    log_metrics: bool
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

    def asdict(self):
        return {
            "max_epochs": self.max_epochs,
            "warmup_epochs": self.warmup_epochs,
            "accelerator": self.accelerator,
            "seed": self.seed,
            "criterion": self.criterion.asdict(),
            "optimizer": self.optimizer.asdict(),
            "scheduler": self.scheduler.asdict(),
            "early_stopping": self.early_stopping.asdict(),
        }
