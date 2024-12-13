import torch


def calculate_dice_score(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 1e-6
) -> float:
    """
    Calculates Dice Coefficient between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.
        epsilon (float): Small value to avoid division by zero. Default is 1e-6.

    Returns:
        float: Dice Coefficient value.
    """
    # Flatten the tensors
    predictions_flatten = predictions.view(-1)
    targets_flatten = targets.view(-1)

    # Ensure EPSILON is on the same device and data type as predictions
    epsilon = torch.tensor(epsilon, device=predictions.device, dtype=predictions.dtype)

    # Calculate intersection and union
    intersection = (predictions_flatten * targets_flatten).sum()
    dice = (2.0 * intersection + epsilon) / (predictions_flatten.sum() + targets_flatten.sum() + epsilon)

    return dice.item()