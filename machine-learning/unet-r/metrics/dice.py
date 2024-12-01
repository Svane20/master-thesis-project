import torch


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute the Dice coefficient

    Args:
        preds (torch.Tensor): Predictions
        targets (torch.Tensor): Targets
        epsilon (float): Smoothing factor to avoid division by zero. Default: 1e-6

    Returns:
        float: Dice coefficient
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice.mean().item()
