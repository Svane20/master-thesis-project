import torch
from PIL import Image

import os

from constants.directories import DATA_TEST_DIRECTORY
from constants.hyperparameters import BATCH_SIZE
from constants.outputs import TRAINED_MODEL_CHECKPOINT_NAME
from dataset.data_loaders import create_test_data_loader
from dataset.transforms import get_test_transforms
from model.unet import UNetV0
from testing.inference import evaluate_model, predict_image
from testing.visualization import save_predictions, save_prediction, remove_background
from utils.checkpoints import load_model_checkpoint
from utils.device import get_device, get_torch_compile_backend

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def main():
    # Get test transforms
    transform = get_test_transforms()

    # Setup device
    device = get_device()

    # Load trained model
    model = UNetV0(in_channels=3, out_channels=1, dropout=0.5)
    model, _ = load_model_checkpoint(model=model, model_name=TRAINED_MODEL_CHECKPOINT_NAME, device=device)
    model = torch.compile(model, backend=get_torch_compile_backend())

    # Model evaluation
    test_data_loader = create_test_data_loader(
        directory=DATA_TEST_DIRECTORY,
        batch_size=BATCH_SIZE,
        transform=transform,
        num_workers=os.cpu_count() - 1,
        pin_memory=True
    )
    evaluate_model(model, test_data_loader, device)
    save_predictions(model, test_data_loader, device, num_batches=1)

    # Single image evaluation
    image_path = DATA_TEST_DIRECTORY / "images" / "cf89c3220bc4_03.jpg"
    image = Image.open(image_path).convert("RGB")

    predicted_mask = predict_image(image, model, transform, device)
    save_prediction(image, predicted_mask)
    remove_background(image, predicted_mask)


if __name__ == "__main__":
    main()
