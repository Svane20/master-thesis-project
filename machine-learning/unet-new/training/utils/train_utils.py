from pathlib import Path

import torch

import logging
from typing import Optional
import random
import numpy as np


class Phase:
    TRAIN = "train"
    VAL = "val"


class DurationMeter:
    """
    Duration meter to measure the time taken for each phase.
    """

    def __init__(self, name: str, device: torch.device, fmt: str = ":f"):
        """
        Args:
            name (str): Name of the meter.
            device (str): Device to run the meter on.
            fmt (str): Format of the meter. Default is ":f".
        """
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self) -> None:
        """
        Reset the meter.
        """
        self.val = 0

    def update(self, val: float) -> None:
        """
        Update the meter.
        """
        self.val = val

    def add(self, val: float) -> None:
        """
        Add the value to the meter.

        Args:
            val (float): Value to add.
        """
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


def human_readable_time(time_seconds: float) -> str:
    """
    Convert seconds to human-readable time.

    Args:
        time_seconds (float): Time in seconds.

    Returns:
        str: Human readable time.
    """
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    return f"{days:02}d {hours:02}h {minutes:02}m"


def get_amp_type(amp_type: Optional[str] = None) -> Optional[torch.dtype]:
    """
    Get the AMP type.

    Args:
        amp_type (str): AMP type. Default is None.

    Returns:
        torch.dtype: AMP type.
    """
    if amp_type is None:
        return None

    assert amp_type in ["bfloat16", "float16"], "Invalid Amp type."

    return torch.bfloat16 if amp_type == "bfloat16" else torch.float16


def set_seeds(seed: int) -> None:
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.

    Args:
        seed (int): Random seed.
    """
    logging.info(f"Machine seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_resume_checkpoint(checkpoint_path: str) -> Optional[Path]:
    """
    Get the resume checkpoint path.

    Args:
        checkpoint_path (str): Checkpoint path.

    Returns:
        Optional[str]: Resume checkpoint path.
    """
    if checkpoint_path is None:
        return None

    current_directory = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = current_directory / checkpoint_path
    if not checkpoint_path.is_file():
        return None

    return checkpoint_path
