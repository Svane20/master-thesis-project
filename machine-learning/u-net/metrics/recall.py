import torch

def calculate_recall(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates recall between predictions and targets.

    Args:
        predictions (torch.Tensor): Binary model predictions (0 or 1).
        targets (torch.Tensor): Binary ground truth targets (0 or 1).

    Returns:
        float: Recall value.
    """
    # Flatten the tensors
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    EPSILON = torch.tensor(1e-6, device=predictions.device, dtype=predictions.dtype)

    # True Positives (TP): predictions == 1 & targets == 1
    TP = (predictions_flat * targets_flat).sum()

    # False Negatives (FN): predictions == 0 & targets == 1
    FN = ((1 - predictions_flat) * targets_flat).sum()

    recall = (TP + EPSILON) / (TP + FN + EPSILON)

    return recall.item()