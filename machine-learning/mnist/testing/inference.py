import torch
from torchvision.transforms import transforms

from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from typing import Tuple


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> torch.Tensor:
    """
    Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation

    Returns:
        torch.Tensor: Predictions
    """
    y_preds = []

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Making predictions"):
            y_logits = model(X.to(device))

            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

            y_preds.append(y_pred.cpu())

    return torch.cat(y_preds)


def predict(
        model: torch.nn.Module,
        image: Image.Image,
        device: torch.device,
        transform: transforms.Compose = None,
) -> Tuple[int, np.ndarray]:
    """
    Predict the class of an image.

    Args:
        model (torch.nn.Module): Model to use for prediction
        image (torch.Tensor): Image to predict the class for
        transform (transforms.Compose): Transform to apply to the image
        device (torch.device): Device to use for prediction

    Returns:
        Tuple[int, torch.Tensor]: Predicted class and probabilities
    """
    # Preprocess the image
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    transformed_image = image_transform(image).unsqueeze(dim=0)

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()


    with torch.inference_mode():
        pred = model(transformed_image.to(device))

    probabilities = torch.softmax(pred, dim=1).cpu().squeeze().numpy()
    label = int(probabilities.argmax())

    return label, probabilities
