from pydantic import BaseModel, Field
from torch import Tensor

import logging
import sys
from typing import Dict, Union
import wandb
from numpy import ndarray

from unet.configuration.training.base import LoggingConfig, WandbConfig


class WeightAndBiasesConfig(BaseModel):
    """
    Configuration class for Weights & Biases initialization.
    """
    epochs: int = Field(description="Number of training epochs")
    learning_rate: float = Field(description="Learning rate for the optimizer")
    learning_rate_decay: float = Field(description="Learning rate decay for the optimizer")
    seed: int = Field(description="Random seed for reproducibility")
    device: str = Field(description="Device type, e.g., 'cuda' or 'cpu'")


Scalar = Union[Tensor, ndarray, int, float]


class WandbLogger:
    def __init__(self, configuration: WandbConfig, wandb_configuration: WeightAndBiasesConfig):
        self.configuration = configuration
        self.enabled = configuration.enabled

        if not self.enabled:
            self.run = None
            return

        self.run = wandb.init(
            project=configuration.project,
            entity=configuration.entity,
            tags=configuration.tags,
            notes=configuration.notes,
            group=configuration.group,
            job_type=configuration.job_type,
            config=wandb_configuration.model_dump()
        )

    def log(self, name: str, payload: Scalar, step: int) -> None:
        """
        Log the payload to the wandb.

        Args:
            name (str): Name of the log.
            payload (Scalar): Payload to log.
            step (int): Step to log.
        """
        if not self.run:
            return

        self.run.log({name: payload}, step=step)

    def log_dict(self, payload: Dict[str, float], step: int) -> None:
        """
        Log the dictionary to the wandb.

        Args:
            payload (Dict[str, float]): Payload to log.
            step (int): Step to log.
        """
        if not self.run:
            return

        log_data = {**payload}
        self.run.log(log_data, step=step)

    def finish(self) -> None:
        """
        Finish the wandb run.
        """
        if not self.run:
            return

        self.run.finish()


class Logger:
    def __init__(self, logging_configuration: LoggingConfig, wandb_configuration: WeightAndBiasesConfig):
        configuration = logging_configuration.wandb

        self.wandb_logger = WandbLogger(configuration, wandb_configuration)

    def log(self, name: str, payload: Scalar, step: int) -> None:
        """
        Log the data to the logger.

        Args:
            name (str): Name of the log.
            payload (Scalar): Data to log.
            step (int): Step to log.
        """
        if self.wandb_logger:
            self.wandb_logger.log(name, payload, step)

    def log_dict(self, payload: Dict[str, float], step: int) -> None:
        """
        Log the dictionary to the logger.

        Args:
            payload (Dict[str, float]): Data to log.
            step (int): Step to log.
        """
        if self.wandb_logger:
            self.wandb_logger.log_dict(payload, step)

    def finish(self) -> None:
        """
        Finish the logger.
        """
        if self.wandb_logger:
            self.wandb_logger.finish()


def setup_logging(name: str) -> None:
    """
    Setup logging for the training process.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel("INFO")

    # Create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # Set up the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    logger.addHandler(console_handler)

    logging.root = logger


def shutdown_logging() -> None:
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers

    for handler in handlers:
        handler.close()
