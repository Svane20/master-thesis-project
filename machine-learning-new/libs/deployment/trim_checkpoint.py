import torch
import torch.nn as nn

from pathlib import Path


def save_model_checkpoint(
        model: nn.Module,
        device: torch.device,
        model_name: str,
        directory: Path,
):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): Model.
        device (torch.device): Device.
        model_name (str): Model name.
        directory (Path): Directory to save the model checkpoint to.
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{model_name}.pt"

    # Set model to device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Save the model checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Minimal model checkpoint saved at {checkpoint_path}")
