import torch

def calculate_precision(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates precision between predictions and targets.

    Args:
        predictions (torch.Tensor): Binary model predictions (0 or 1).
        targets (torch.Tensor): Binary ground truth targets (0 or 1).

    Returns:
        float: Precision value.
    """
    # Flatten the tensors
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    EPSILON = torch.tensor(1e-6, device=predictions.device, dtype=predictions.dtype)

    # True Positives (TP): predictions == 1 & targets == 1
    TP = (predictions_flat * targets_flat).sum()

    # False Positives (FP): predictions == 1 & targets == 0
    FP = (predictions_flat * (1 - targets_flat)).sum()

    precision = (TP + EPSILON) / (TP + FP + EPSILON)

    return precision.item()