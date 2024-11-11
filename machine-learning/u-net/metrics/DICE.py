import torch


def calculate_DICE(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Dice Coefficient between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.

    Returns:
        float: Dice Coefficient value.
    """
    # Flatten the tensors
    predictions_flatten = predictions.view(-1)
    targets_flatten = targets.view(-1)

    # Ensure EPSILON is on the same device and data type as predictions
    EPSILON = torch.tensor(1e-6, device=predictions.device, dtype=predictions.dtype)

    # Calculate intersection and union
    intersection = (predictions_flatten * targets_flatten).sum()
    dice = (2.0 * intersection + EPSILON) / (predictions_flatten.sum() + targets_flatten.sum() + EPSILON)

    return dice.item()
