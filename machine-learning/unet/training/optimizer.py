import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW, SGD

from typing import Type

from unet.configuration.training import OptimizerConfig


def construct_optimizer(model: nn.Module, config: OptimizerConfig) -> optim.Optimizer:
    """
    Construct the optimizer for the model.

    Args:
        model (nn.Module): Model to train.
        config (OptimizerConfig): Configuration for the optimizer.

    Returns:
        optim.Optimizer: Optimizer for the model.
    """
    # Use the helper function to get the PyTorch optimizer class
    optimizer_cls = _get_optimizer_by_name(config.name)

    # Instantiate the optimizer with the given learning rate
    optimizer = optimizer_cls(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def _get_optimizer_by_name(name: str) -> Type[Adam | AdamW | SGD]:
    """
    Get the optimizer class by name.

    Args:
        name (str): Name of the optimizer.

    Returns:
        optim.Optimizer: Optimizer class.
    """
    match name:
        case "Adam":
            return optim.Adam
        case "AdamW":
            return optim.AdamW
        case "SGD":
            return optim.SGD
        case _:
            raise ValueError(f"Unsupported optimizer: {name}")
