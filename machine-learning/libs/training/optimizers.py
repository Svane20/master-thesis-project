import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW, SGD

from typing import Type

from ..configuration.training.optimizer import OptimizerConfig


class GradientClipper:
    """
    Gradient clipper to clip the gradients of the model.
    """

    def __init__(self, enabled: bool = True, max_norm: float = 1.0, norm_type: int = 2) -> None:
        """
        Args:
            max_norm (float): Maximum norm of the gradients. Default is 1.0.
            norm_type (int): Type of the norm. Default is 2.
        """
        assert isinstance(max_norm, (int, float)) or max_norm is None

        self.enabled = enabled
        self.max_norm = max_norm if max_norm is None else float(max_norm)
        self.norm_type = norm_type

    def __call__(self, model: nn.Module) -> None:
        if not self.enabled:
            return

        if self.max_norm is None:
            return

        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
        )


def construct_optimizer(model: nn.Module, config: OptimizerConfig) -> optim.Optimizer:
    """
    Construct the optimizer for the model.

    Args:
        model (nn.Module): Model to train.
        config (OptimizerConfig): Configuration for the optimizer.

    Returns:
        optim.Optimizer: Optimizer for the model.
    """
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
        Type[Adam | AdamW | SGD]: Optimizer class.
    """
    match name:
        case "Adam":
            return optim.Adam
        case "AdamW":
            return optim.AdamW
        case "SGD":
            return optim.SGD
        case _:
            raise ValueError(f"Optimizer: {name} is not supported.")
