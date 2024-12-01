import torch

from tqdm.auto import tqdm
from PIL import Image
import albumentations as A
import numpy as np

from metrics.DICE import calculate_DICE, calculate_DICE_edge


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> None:
    """
    Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
    """
    model.eval()

    total_dice, total_dice_edge = 0, 0
    num_batches = 0

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            num_batches += 1

            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()

            total_dice += calculate_DICE(preds, y)
            total_dice_edge += calculate_DICE_edge(preds, y)

    # Compute average metrics
    avg_dice = total_dice / num_batches
    avg_dice_edge = total_dice_edge / num_batches

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Dice Edge Score: {avg_dice_edge:.4f}")


def predict_image(
        image: Image.Image,
        model: torch.nn.Module,
        transform: A.Compose,
        device: torch.device,
) -> np.ndarray:
    """
    Predict the binary mask for a single image.

    Args:
        image (Image.Image): Input image
        model (torch.nn.Module): Model to use for prediction
        transform (albumentations.Compose): Transform to apply to the image
        device (torch.device): Device to use for inference

    Returns:
        np.ndarray: Predicted binary mask
    """
    model.eval()

    # Convert PIL image to NumPy array and apply transforms
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

    with torch.inference_mode():
        y_logits = model(image_tensor)
        y_preds = torch.sigmoid(y_logits)
        preds = (y_preds > 0.5).float()  # Binary mask

    return preds.squeeze(0).cpu().numpy()  # Remove batch and convert to NumPy
