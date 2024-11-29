import torch

import wandb
from typing import Any, Dict, Optional, Tuple, Literal
from pathlib import Path

from constants.directories import CHECKPOINTS_DIRECTORY
from training.early_stopping import EarlyStopping


def save_checkpoint(
        model: torch.nn.Module,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        directory: Path = CHECKPOINTS_DIRECTORY,
        postfix: str = 'latest',
        extension: Literal["pth", "pt"] = "pth"
) -> Path:
    """
    Saves a model checkpoint to the specified directory.

    Args:
        model (torch.nn.Module): Model to save.
        model_name (str): Name of the model.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        metrics (Dict[str, Any], optional): Metrics to save.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to save. Default is None.
        early_stopping (EarlyStopping, optional): Early stopping to save. Default is None.
        directory (Path): Directory to save the model to. Default is CHECKPOINTS_DIRECTORY.
        postfix (str): Postfix to add to the saved model checkpoint. Default is "latest".
        extension (str): Extension to use. Default is ".pth".

    Returns:
        pathlib.Path: Path to the saved model checkpoint.
    """
    # Ensure the directory exists
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}_{postfix}.{extension}"

    print(f"[INFO] Saving model checkpoint to {save_path}")

    # Construct the checkpoint dictionary dynamically
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if early_stopping:
        checkpoint["early_stopping_state_dict"] = early_stopping.state_dict()

    if metrics:
        checkpoint.update(metrics)

    # Save the checkpoint
    torch.save(obj=checkpoint, f=save_path)

    # Log success
    if metrics and "epoch" in metrics and "loss" in metrics:
        print(f"[INFO] Best model saved at epoch {metrics['epoch']} with val_loss: {metrics['loss']:.4f}")
    else:
        print("[INFO] Model checkpoint saved without metrics.")

    return save_path


def save_checkpoint_to_wandb(
        run: wandb.sdk.wandb_run.Run,
        model: torch.nn.Module,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        directory: Path = CHECKPOINTS_DIRECTORY,
        postfix: str = 'latest',
        extension: Literal["pth", "pt"] = "pth"
) -> None:
    """
    Save a checkpoint to Weights & Biases.

    Args:
        run (wandb.sdk.wandb_run.Run): Weights & Biases run.
        model (torch.nn.Module): Model to save.
        model_name (str): Name of the model.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        metrics (Dict[str, Any], optional): Metrics to save.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to save. Default is None.
        early_stopping (EarlyStopping, optional): Early stopping to save. Default is None.
        directory (Path): Directory to save the model to. Default is CHECKPOINTS_DIRECTORY.
        postfix (str): Postfix to add to the saved model checkpoint. Default is "latest".
        extension (str): Extension to use. Default is ".pth".
    """
    # Create the artifact
    artifact = wandb.Artifact(
        name=model_name,
        type='model',
        metadata=dict(metrics)
    )

    # Save the checkpoint
    checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        metrics=metrics,
        model_name=model_name,
        directory=directory,
        extension=extension,
        postfix=postfix
    )

    # Add the checkpoint file to the artifact
    artifact.add_file(str(checkpoint_path))

    # Save the artifact
    run.save(glob_str=str(checkpoint_path), base_path=str(directory))

    # Log the artifact to wandb
    run.log_artifact(artifact, aliases=['best'])


def load_checkpoint(
        model: torch.nn.Module,
        model_name: str,
        device: torch.device,
        directory: Path = CHECKPOINTS_DIRECTORY,
        extension: str = "pth",
        postfix: str = 'latest',
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_is_compiled: bool = True,
) -> Tuple[
    torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler], Optional[EarlyStopping], Dict[
        str, Any]]:
    """
    Loads the model, optimizer, and scheduler states from a checkpoint file.

    Args:
        model (torch.nn.Module): Model to load the weights into.
        model_name (str): Name of the model.
        directory (Path): Directory to load the model checkpoint from. Default is "checkpoints".
        extension (str): Extension to use. Default is ".pth".
        postfix (str): Postfix to add to the saved model checkpoint. Default is "latest".
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore the state. Default is None.
        scheduler (any, optional): Scheduler to restore the state. Default is None.
        early_stopping (EarlyStopping, optional): Early stopping to restore the state. Default is None.
        device (torch.device): Device to load the model onto.
        checkpoint_is_compiled (bool): Whether the checkpoint is compiled. Default is True.

    Returns:
        Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler], Optional[torch.optim.lr_scheduler], Optional[EarlyStopping], Dict[str, Any]]:
            - Model with loaded weights.
            - Optimizer with loaded state (if provided).
            - Scheduler with loaded state (if provided).
            - Warmup scheduler with loaded state (if provided).
            - Early stopping with loaded state (if provided).
            - Dictionary containing additional info such as epoch, loss, and accuracy.
    """
    model_file = directory / f"{model_name}_{postfix}.{extension}"
    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {model_file}")

    # Load the checkpoint
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    # Handle `_orig_mod.` prefix in checkpoint keys if it was saved from a compiled model
    state_dict = checkpoint["model_state_dict"]
    if checkpoint_is_compiled:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load model state
    model.load_state_dict(state_dict)
    model.to(device)

    model.train()

    # Load optimizer state if provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if early_stopping and "early_stopping_state_dict" in checkpoint:
        early_stopping.load_state_dict(checkpoint["early_stopping_state_dict"])

    # Additional checkpoint information
    checkpoint_info = {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        "dice": checkpoint.get("dice"),
        "dice_edge": checkpoint.get("dice_edge"),
    }

    return model, optimizer, scheduler, early_stopping, {**checkpoint_info}


def load_model_checkpoint(
        model: torch.nn.Module,
        model_name: str,
        device: torch.device,
        directory: Path = CHECKPOINTS_DIRECTORY,
        extension: str = "pth",
        postfix: str = 'latest',
        checkpoint_is_compiled: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the model checkpoint.

    Args:
        model (torch.nn.Module): Model to load the weights into.
        model_name (str): Name of the model.
        device (torch.device): Device to load the model onto.
        directory (Path): Directory to load the model checkpoint from. Default is "checkpoints".
        extension (str): Extension to use. Default is ".pth".
        postfix (str): Postfix to add to the saved model checkpoint. Default is "latest".
        checkpoint_is_compiled (bool): Whether the checkpoint is compiled. Default is True.

    Returns:
        Tuple[torch.nn.Module, Dict[str, Any]]: Model with loaded weights and additional checkpoint information.
    """
    model_file = directory / f"{model_name}_{postfix}.{extension}"
    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {model_file}")

    # Load the checkpoint
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    # Handle `_orig_mod.` prefix in checkpoint keys if it was saved from a compiled model
    state_dict = checkpoint["model_state_dict"]
    if checkpoint_is_compiled:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load model state
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()

    # Additional checkpoint information
    checkpoint_info = {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        "dice": checkpoint.get("dice"),
        "iou": checkpoint.get("iou"),
    }

    return model, {**checkpoint_info}


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
