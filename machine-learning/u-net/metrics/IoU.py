import torch


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

    # Ensure EPSILON is on the same device and data type as predictions
    EPSILON = torch.tensor(1e-6, device=predictions.device, dtype=predictions.dtype)

    # Calculate intersection and union
    intersection = (predictions_flatten * targets_flatten).sum()
    union = predictions_flatten.sum() + targets_flatten.sum() - intersection

    # Calculate IoU
    iou = (intersection + EPSILON) / (union + EPSILON)  # Add epsilon to prevent division by zero

    return iou.item()
