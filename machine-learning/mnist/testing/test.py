import torch
from PIL import Image
from torchvision.transforms import transforms

from typing import List, Tuple, Optional, Dict, Any
import os

from dataset.data_loader import create_test_data_loader
from dataset.mnist_dataset import get_dataset
from model.mnist import FashionMnistModelV0
from testing.inference import evaluate_model, predict
from testing.visualization import plot_confusion_matrix, plot_prediction
from utils import load_checkpoint, get_device

SEED: int = 42
BATCH_SIZE: int = 32
MODEL_NAME: str = "FashionMNISTModelV0"


def load_model(
        classes: List[str],
        target_device: torch.device
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict[str, Any]]:
    loaded_model = FashionMnistModelV0(
        input_shape=1,
        hidden_units=32,
        output_shape=len(classes)
    )

    return load_checkpoint(model=loaded_model, model_name=MODEL_NAME, device=target_device)


if __name__ == "__main__":
    # Get test data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_data = get_dataset(train=False, transform=transform, )
    test_data_loader = create_test_data_loader(
        test_data,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count() if torch.cuda.is_available() else 2,
    )

    # Setup device
    device = get_device()

    class_names = test_data.classes

    # Load trained model
    model, _, _, _ = load_model(class_names, device)

    # Make predictions
    y_preds = evaluate_model(model, test_data_loader, device)

    # Plot confusion matrix
    plot_confusion_matrix(
        class_names,
        y_preds,
        targets=test_data.targets,
        task="multiclass"
    )

    # Get a sample image
    test_image = test_data.data[0]
    test_image = Image.fromarray(test_image.numpy(), mode='L')

    # Make a prediction on the sample image
    label, probabilities = predict(model, test_image, device, transform)
    plot_prediction(class_names, test_image, label, probabilities)
