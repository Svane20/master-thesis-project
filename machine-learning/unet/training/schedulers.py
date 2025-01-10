import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, ExponentialLR, \
    LRScheduler

from typing import Optional, cast, Literal

from configuration.training.scheduler import SchedulerConfig


class SchedulerWrapper:
    """
    A wrapper class that constructs a scheduler from a given config,
    and provides a unified .step() method handling different scheduler types.
    """

    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig) -> None:
        """
        Args:
            optimizer (optim.Optimizer): The optimizer to step.
            config (SchedulerConfig): The scheduler configuration.
        """
        self.optimizer = optimizer
        self.config = config
        self.scheduler = self._construct_scheduler()

    def step(self, metric: Optional[float] = None) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires a metric to step.")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _construct_scheduler(self) -> Optional[LRScheduler]:
        if not self.config.enabled:
            return None

        match self.config.name:
            case "CosineAnnealingLR":
                return CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.config.parameters["t_max"],
                    eta_min=self.config.parameters.get("eta_min", 0)
                )
            case "ReduceLROnPlateau":
                raw_mode = self.config.parameters.get("mode", "min")
                if raw_mode not in ("min", "max"):
                    raise ValueError(f"'mode' must be either 'min' or 'max', got {raw_mode}.")

                mode_literal = cast(Literal["min", "max"], raw_mode)

                return ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    mode=mode_literal,
                    factor=self.config.parameters.get("factor", 0.1),
                    patience=self.config.parameters.get("patience", 10),
                    threshold=self.config.parameters.get("threshold", 1e-4),
                    cooldown=self.config.parameters.get("cooldown", 0),
                    min_lr=self.config.parameters.get("min_lr", 0),
                    verbose=self.config.parameters.get("verbose", False),
                )
            case "StepLR":
                return StepLR(
                    optimizer=self.optimizer,
                    step_size=self.config.parameters["step_size"],
                    gamma=self.config.parameters.get("gamma", 0.1)
                )
            case "ExponentialLR":
                return ExponentialLR(
                    optimizer=self.optimizer,
                    gamma=self.config.parameters.get("gamma", 0.9)
                )
            case _:
                raise ValueError(f"Scheduler {self.config.name} is not supported.")
