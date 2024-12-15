import torch

from torchinfo import summary
from typing import Tuple


def get_model_summary(
        model: torch.nn.Module,
        input_size: Tuple[int, int, int, int],
        column_names=None,
        col_width: int = 20,
        row_settings=None
) -> None:
    """
    Prints the model summary.

    Args:
        model (torch.nn.Module): Model to print the summary for.
        input_size: Tuple[int, int, int, int]: Input size (batch_size, colour channels, height, width) for the model.
        column_names (List[str]): Column names for the summary. Default is ["input_size", "output_size", "num_params", "trainable"].
        col_width (int): Column width for the summary. Default is 20.
        row_settings (List[str]): Row settings for the summary. Default is ["var_names"].
    """
    if column_names is None:
        column_names = ["input_size", "output_size", "num_params", "trainable"]

    if row_settings is None:
        row_settings = ["var_names"]

    summary(
        model=model,
        input_size=input_size,
        col_names=column_names,
        col_width=col_width,
        row_settings=row_settings
    )
