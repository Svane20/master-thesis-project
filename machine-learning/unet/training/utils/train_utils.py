import torch
import torch.distributed as dist

import logging
from typing import Optional, Tuple, List, Union, Dict
import random
import numpy as np
from pathlib import Path
import os
from datetime import timedelta


class Phase:
    TRAIN = "train"
    VAL = "val"


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, device: str, fmt: str = ":f"):
        """
        Args:
            name (str): Name of the meter.
            device (str): Device to run the meter on.
            fmt (str): Format of the meter. Default is ":f".
        """
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val: float, n: int = 1):
        """
        Update the meter.

        Args:
            val (float): Value to update.
            n (int): Number of iterations. Default is 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"

        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Computes and stores the current, avg, and max of peak Mem usage per iteration"""

    def __init__(self, name, device, fmt=":f"):
        """
        Args:
            name (str): Name of the meter.
            device (str): Device to run the meter on.
            fmt (str): Format of the meter. Default is ":f".
        """
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0  # Per iteration max usage
        self.avg = 0  # Avg per iteration max usage
        self.peak = 0  # Peak usage for lifetime of program
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, n: int = 1, reset_peak_usage: bool = True):
        """
        Update the meter.

        Args:
            n (int): Number of iterations.
            reset_peak_usage (bool): Reset the peak usage. Default is True.
        """
        self.val = torch.cuda.max_memory_allocated() // 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)

        if reset_peak_usage:
            torch.cuda.reset_peak_memory_stats()

    def __str__(self):
        fmtstr = (
                "{name}: {val"
                + self.fmt
                + "} ({avg"
                + self.fmt
                + "}/{peak"
                + self.fmt
                + "})"
        )

        return fmtstr.format(**self.__dict__)


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

        Args:
            val (float): Value to update.
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


Meter = Union[AverageMeter, DurationMeter, MemMeter]


class ProgressMeter:
    """
    Progress meter to display the progress of the training.
    """

    def __init__(
            self,
            num_batches: int,
            meters: List[Meter],
            real_meters: Dict[str, Meter],
            prefix: str = ""
    ):
        """
        Args:
            num_batches (int): Number of batches.
            meters (List[Meter]): List of meters.
            real_meters (Dict[str, Meter]): Real meters.
            prefix (str): Prefix for the meter. Default is "".
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch: int, enable_print: bool = False) -> None:
        """
        Display the progress of the training.

        Args:
            batch (int): Batch number.
            enable_print (bool): Enable print. Default is False.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [
                    f"{os.path.join(name, subname)}: {val:.4f}"
                    for subname, val in meter.compute().items()
                ]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))

        if enable_print:
            print(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """
        Get the batch format string.

        Args:
            num_batches (int): Number of batches.

        Returns:
            str: Batch format string
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"

        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def setup_distributed_backend(backend, timeout_mins):
    """
    Initialize torch.distributed and set the CUDA device.
    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environ variable "LOCAL_RANK" which is used to set the CUDA device.
    """
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logging.info(f"Setting up torch.distributed with a timeout of {timeout_mins} mins")

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_mins))

    return dist.get_rank()


def get_machine_local_and_dist_rank() -> Tuple[int, int]:
    """
    Get the distributed and local rank of the current gpu.

    Returns:
        Tuple[int, int]: Local rank and distributed rank
    """
    local_rank = int(os.environ.get("LOCAL_RANK", None))
    distributed_rank = int(os.environ.get("RANK", None))
    assert (
            local_rank is not None and distributed_rank is not None
    ), "Please the set the RANK and LOCAL_RANK environment variables."

    return local_rank, distributed_rank


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

    assert amp_type in ["bfloat16", "float16"], f"Invalid AMP type: {amp_type}. Choose 'bfloat16' or 'float16'."

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


def makedir(directory_path: str) -> bool:
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): Directory path.

    Returns:
        bool: Whether the directory was created successfully.
    """
    is_success = False

    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        is_success = True
    except BaseException as e:
        logging.error(f"Error creating directory: {directory_path}", e)

    return is_success


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

    root_directory = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = root_directory / checkpoint_path
    if not checkpoint_path.is_file():
        return None

    return checkpoint_path


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
