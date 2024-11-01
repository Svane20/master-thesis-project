import torch

from typing import List
from pathlib import Path

from constants.directories import MODELS_DIRECTORY
from dataset.data_loader import create_train_data_loader
from dataset.mnist_dataset import get_test_data
from model.mnist import FashionMnistModelV0
from testing.inference import evaluate_model
from testing.visualization import plot_confusion_matrix
from utils import load_trained_model, get_device

SEED: int = 42
BATCH_SIZE: int = 32
MODELS_PATH: Path = MODELS_DIRECTORY / "FashionMNISTModelV0.pth"


def load_model(classes: List[str], target_device: torch.device):
    loaded_model = FashionMnistModelV0(
        input_shape=1,
        hidden_units=10,
        output_shape=len(classes)
    )
    loaded_model = load_trained_model(loaded_model, MODELS_PATH).to(target_device)

    return loaded_model


if __name__ == "__main__":
    # Get test data loader
    test_data = get_test_data()
    test_data_loader = create_train_data_loader(test_data, batch_size=BATCH_SIZE)

    # Setup device
    device = get_device()

    class_names = test_data.classes

    # Load trained model
    model = load_model(class_names, device)

    # Make predictions
    y_preds = evaluate_model(model, test_data_loader, device)

    # Plot confusion matrix
    plot_confusion_matrix(
        class_names,
        y_preds,
        targets=test_data.targets,
        task="multiclass"
    )
