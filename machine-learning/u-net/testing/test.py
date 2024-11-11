import torch
from torch.utils.data import DataLoader

import os
from typing import Tuple, Optional, Any, Dict

from constants.directories import DATA_TEST_DIRECTORY
from constants.hyperparameters import BATCH_SIZE
from constants.outputs import MODEL_NAME
from dataset.data_loaders import create_test_data_loader
from dataset.transforms import get_test_transforms
from model.unet import UNetV0
from testing.inference import evaluate_model
from testing.visualization import save_predictions
from utils import load_checkpoint, get_device

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def load_model(
        target_device: torch.device
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict[str, Any]]:
    """
    Load the model.

    Args:
        target_device (torch.device): Device to load the model on.

    Returns:
        Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict[str, Any]]: Model, optimizer, scheduler, and checkpoint.
    """
    loaded_model = UNetV0(
        in_channels=3,
        out_channels=1,
    )

    return load_checkpoint(model=loaded_model, model_name=MODEL_NAME, device=target_device)


def get_data_loaders(batch_size: int) -> DataLoader:
    """
    Get the test data loader.

    Args:
        batch_size (int): Batch size for the data loaders.

    Returns:
        DataLoader: Test data loader.
    """
    transform = get_test_transforms()

    return create_test_data_loader(
        directory=DATA_TEST_DIRECTORY,
        batch_size=batch_size,
        transform=transform,
        num_workers=os.cpu_count() if torch.cuda.is_available() else 2,
    )


def main():
    # Get test data loader
    test_data_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # Setup device
    device = get_device()

    # Load trained model
    model, _, _, _ = load_model(device)
    model.to(device)

    # Make predictions
    # evaluate_model(model, test_data_loader, device)

    # Save the predictions
    save_predictions(model, test_data_loader, device, num_batches=1)


if __name__ == "__main__":
    main()
