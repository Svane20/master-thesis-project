import torch

from utils.edge_detection import EPSILON


def calculate_IoU(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Intersection over Union (IoU) between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.

    Returns:
        float: Intersection over Union (IoU) value.
    """
    # Flatten the tensors
    predictions_flatten = predictions.view(-1)
    targets_flatten = targets.view(-1)

    # Calculate intersection and union
    intersection = (predictions_flatten * targets_flatten).sum()
    union = predictions_flatten.sum() + targets_flatten.sum() - intersection

    # Ensure EPSILON is on the same device and data type as predictions
    epsilon = torch.tensor(EPSILON, device=predictions.device, dtype=predictions.dtype)

    # Calculate IoU
    iou = (intersection + epsilon) / (union + epsilon)  # Add epsilon to prevent division by zero

    return iou.item()
