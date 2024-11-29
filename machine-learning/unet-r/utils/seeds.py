import torch
import numpy as np
import random


def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Random seed to set. Default is 42.
    """
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module
    torch.manual_seed(seed)  # Set seed for PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for CUDA
        torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
