import torch

import wandb
from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path

from constants.directories import CHECKPOINTS_DIRECTORY


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epoch: int,
        loss: float,
        model_name: str,
        directory: Path = CHECKPOINTS_DIRECTORY,
        extension: str = "pth"
) -> Path:
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

    Returns:
        pathlib.Path: Path to the saved model checkpoint.
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

    return save_path


def save_checkpoint_to_wandb(
        run: wandb.sdk.wandb_run.Run,
        model_name: str,
        checkpoint_path: Union[str, Path],
        epoch: int,
        dice_score: float,
        loss: float
) -> None:
    """
    Save a checkpoint to Weights & Biases.

    Args:
        run (wandb.sdk.wandb_run.Run): Weights & Biases run.
        model_name (str): Name of the model.
        checkpoint_path (Union[str, Path]): Path to the checkpoint file.
        epoch (int): Current epoch number.
        dice_score (float): Test DICE score.
        loss (float): Test loss.
    """
    # Create the artifact
    artifact = wandb.Artifact(
        name=model_name,
        type='model',
        metadata={
            'epoch': epoch,
            'dice_score': dice_score,
            'loss': loss,
        }
    )

    # Add the checkpoint file to the artifact
    artifact.add_file(str(checkpoint_path))

    # Log the artifact to wandb
    run.log_artifact(artifact, aliases=['best'])


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
        directory (Path): Directory to load the model checkpoint from. Default is "checkpoints".
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


def load_checkpoint_from_wandb(
        model_name: str,
        entity: str,
        project: str,
        run_id: Optional[str] = None,
        artifact_alias: str = "best",
) -> Path:
    """
    Loads the model, optimizer, and scheduler states from a wandb artifact.

    Args:
        model_name (str): Name of the model (used in artifact naming).
        entity (str): wandb entity (username or team name).
        project (str): wandb project name.
        run_id (str, optional): wandb run ID. If provided, will load the artifact from this run. Default is None.
        artifact_alias (str): Alias of the artifact to download. Default is "best".

    Returns:
        pathlib.Path: Path to the downloaded artifact.
    """
    # Initialize wandb API
    api = wandb.Api()

    # Construct the artifact path
    if run_id:
        # If run_id is provided, get the artifact from that run
        artifact_path = f'{entity}/{project}/run-{run_id}/{model_name}:{artifact_alias}'
    else:
        # Otherwise, get the artifact directly from the project
        artifact_path = f'{entity}/{project}/{model_name}:{artifact_alias}'

    # Get the artifact
    artifact = api.artifact(artifact_path, type='model')

    # Download the artifact
    artifact_dir = artifact.download()

    return Path(artifact_dir)
