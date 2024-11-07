import torch
from torchinfo import summary

from pathlib import Path
from typing import Tuple, List

from constants.directories import MODELS_DIRECTORY


def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Random seed to set. Default is 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device() -> torch.device:
    """
    Returns the device to run the training on.

    Returns:
        torch.device: Device to run the training on.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def save_model(
        model: torch.nn.Module,
        model_name: str,
        directory: Path = MODELS_DIRECTORY,
        extension: str = "pth"
) -> None:
    """
    Saves a model to the specified directory.

    Args:
        model (torch.nn.Module): Model to save.
        model_name (str): Name of the model.
        directory (str): Directory to save the model to. Default is "models".
        extension (str): Extension to use. Default is ".pth".
    """
    directory.mkdir(parents=True, exist_ok=True)

    assert extension in ["pth", "pt"], "Extension must be either 'pth' or 'pt'"
    save_path = directory / f"{model_name}.{extension}"

    print(f"[INFO] Saving model to {save_path}")
    torch.save(obj=model.state_dict(), f=save_path)


def load_trained_model(
        model: torch.nn.Module,
        model_path: Path,
        device: torch.device
) -> torch.nn.Module:
    """
    Loads a trained model from the specified path.

    Args:
        model (torch.nn.Module): Model to load the trained weights into.
        model_path (str): Path to the trained model.
        device (torch.Device): Device to load the model to.

    Returns:
        torch.nn.Module: Model with trained weights.
    """
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    return model
