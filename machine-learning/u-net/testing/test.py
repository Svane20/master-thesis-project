import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import os
from typing import Tuple, Optional, Any, Dict

from constants.directories import DATA_TEST_DIRECTORY
from dataset.data_loader import create_test_data_loader
from model.unet import UNetV0
from testing.inference import evaluate_model, save_predictions_as_images
from utils import load_checkpoint, get_device

SEED: int = 42
BATCH_SIZE: int = 8
MODEL_NAME: str = "UNetV0"


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
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return create_test_data_loader(
        directory=DATA_TEST_DIRECTORY,
        batch_size=batch_size,
        transform=image_transform,
        target_transform=mask_transform,
        num_workers=os.cpu_count() if torch.cuda.is_available() else 2,
    )


if __name__ == "__main__":
    # Get test data loader
    test_data_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # Setup device
    device = get_device()

    # Load trained model
    model, _, _, _ = load_model(device)

    # Make predictions
    evaluate_model(model, test_data_loader, device)

    # Save the predictions
    save_predictions_as_images(model, test_data_loader, device)

