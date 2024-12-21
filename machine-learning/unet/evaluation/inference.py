import torch

from tqdm.auto import tqdm
from PIL import Image
import albumentations as A
import numpy as np
from typing import Dict

from evaluation.metrics import calculate_dice_score, calculate_dice_edge_score, calculate_iou_score


def _calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics for a batch of predictions and targets.

    Args:
        predictions (torch.Tensor): Predictions.
        targets (torch.Tensor): Ground truth targets.

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    return {
        "dice": calculate_dice_score(predictions, targets),
        "dice_edge": calculate_dice_edge_score(predictions, targets),
        "iou": calculate_iou_score(predictions, targets),
    }


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

    total_metrics = {"dice": 0.0, "dice_edge": 0.0, "iou": 0.0}
    num_batches = 0

    if len(data_loader) == 0:
        print("Warning: DataLoader is empty. No evaluation performed.")
        return

    for X, y in tqdm(data_loader, desc="Evaluating", leave=True):
        num_batches += 1

        # Move data to device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                # Get predictions
                y_logits = model(X)
                y_preds = torch.sigmoid(y_logits)
                preds = (y_preds > 0.5).float()

                # Calculate metrics
                metrics = _calculate_metrics(preds, y)

        # Accumulate metrics
        for key in metrics:
            total_metrics[key] += metrics[key] if metrics[key] is not None else 0.0

    # Compute averages
    avg_metrics = {key: total / num_batches for key, total in total_metrics.items()}

    print(f"Evaluation completed: {avg_metrics}")


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

    # Apply transformations
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            # Get predictions
            y_logits = model(image_tensor)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()  # Binary mask

    return preds.squeeze(0).cpu().numpy()  # Remove batch and convert to NumPy
