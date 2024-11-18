import torch
from PIL import Image

import os
from typing import Tuple, Optional, Any, Dict

from constants.directories import DATA_TEST_DIRECTORY
from constants.hyperparameters import BATCH_SIZE
from constants.outputs import MODEL_OUTPUT_NAME
from dataset.data_loaders import create_test_data_loader
from dataset.transforms import get_test_transforms
from model.unet import UNetV0
from testing.inference import evaluate_model, predict_image
from testing.visualization import save_predictions, save_prediction
from utils.checkpoints import load_checkpoint
from utils.device import get_device

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

    return load_checkpoint(model=loaded_model, model_name=MODEL_OUTPUT_NAME, device=target_device)


def main():
    # Get test transforms
    transform = get_test_transforms()

    # Single image prediction
    image_path = DATA_TEST_DIRECTORY / "images" / "cf89c3220bc4_03.jpg"
    image = Image.open(image_path).convert("RGB")

    # Get test data loader
    test_data_loader = create_test_data_loader(
        directory=DATA_TEST_DIRECTORY,
        batch_size=BATCH_SIZE,
        transform=transform,
        num_workers=os.cpu_count() - 1,
        pin_memory=True
    )

    # Setup device
    device = get_device()

    # Load trained model
    model, _, _, _ = load_model(device)
    model.to(device)

    # Model evaluation
    evaluate_model(model, test_data_loader, device)
    save_predictions(model, test_data_loader, device, num_batches=1)

    # Single image evaluation
    predicted_mask = predict_image(image, model, transform, device)
    save_prediction(image, predicted_mask)


if __name__ == "__main__":
    main()
