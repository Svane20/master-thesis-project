from pydantic import BaseModel, Field
from torch import Tensor

import logging
import sys
from typing import Dict, Union, TextIO
import wandb
from numpy import ndarray
import atexit
import functools

from training.utils.train_utils import makedir
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


def setup_logging(name: str, out_directory: str = None) -> None:
    """
    Setup logging for the training process.

    Args:
        name (str): Name of the logger.
        out_directory (str): Output directory for the logs.

    Returns:
        logging.Logger: Logger object.
    """
    log_filename = None
    if out_directory:
        makedir(out_directory)
        log_filename = f"{out_directory}/log.txt"

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

    if log_filename:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel("INFO")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging() -> None:
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers

    for handler in handlers:
        handler.close()


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename: str) -> TextIO:
    """
    Create a log stream with a buffer size of 10 KB.

    Args:
        filename (str): Name of the log file.

    Returns:
        TextIO: Log stream.
    """
    log_buffer_kb = 10 * 1024  # 10 KB
    io = open(filename, "a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io
