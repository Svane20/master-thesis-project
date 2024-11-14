import torch
from torchinfo import summary

from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from constants.directories import CHECKPOINTS_DIRECTORY


def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Random seed to set. Default is 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epoch: int,
        loss: float,
        model_name: str,
        directory: Path = CHECKPOINTS_DIRECTORY,
        extension: str = "pth"
) -> None:
    """
    Saves a model checkpoint to the specified directory.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to save.
        epoch (int): Epoch number.
        loss (float): Loss value.
        model_name (str): Name of the model.
        directory (str): Directory to save the model to. Default is "models".
        extension (str): Extension to use. Default is ".pth".
    """
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}_best_checkpoint.{extension}"

    print(f"[INFO] Saving model checkpoint to {save_path}")

    torch.save(
        obj={
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            "loss": loss,
        },
        f=save_path
    )

    print(f"[INFO] Best model saved at epoch {epoch + 1} with val_loss: {loss:.4f}")


def load_checkpoint(
        model: torch.nn.Module,
        model_name: str,
        device: torch.device,
        directory: Path = CHECKPOINTS_DIRECTORY,
        extension: str = "pth",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        is_eval: bool = True
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict[str, Any]]:
    """
    Loads the model, optimizer, and scheduler states from a checkpoint file.

    Args:
        model (torch.nn.Module): Model to load the weights into.
        model_name (str): Name of the model.
        directory (Path): Directory to save the model to. Default is "checkpoints".
        extension (str): Extension to use. Default is ".pth".
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore the state. Default is None.
        scheduler (any, optional): Scheduler to restore the state. Default is None.
        device (torch.device): Device to load the model onto.
        is_eval (bool): Set the model to evaluation mode. Default is True.

    Returns:
        Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict[str, Any]]:
            - Model with loaded weights.
            - Optimizer with loaded state (if provided).
            - Scheduler with loaded state (if provided).
            - Dictionary containing additional info such as epoch, loss, and accuracy.
    """
    model_file = directory / f"{model_name}_best_checkpoint.{extension}"
    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {model_file}")

    # Load the checkpoint
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if is_eval:
        model.eval()  # Set to evaluation mode for inference
    else:
        model.train()  # Set to training mode for fine-tuning

    # Load optimizer state if provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Additional checkpoint information
    checkpoint_info = {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
    }

    return model, optimizer, scheduler, checkpoint_info
