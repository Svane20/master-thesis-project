import torch

from utils.edge_detection import compute_edge_map, EPSILON


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
    epsilon = torch.tensor(EPSILON, device=predictions.device, dtype=predictions.dtype)

    # Calculate intersection and union
    intersection = (predictions_flatten * targets_flatten).sum()
    dice = (2.0 * intersection + epsilon) / (predictions_flatten.sum() + targets_flatten.sum() + epsilon)

    return dice.item()


def calculate_DICE_edge(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Dice Coefficient between edge maps of predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.

    Returns:
        float: Dice Coefficient value.
    """
    if predictions.dim() not in [2, 3, 4]:
        raise ValueError("Invalid dimensions for predictions/targets.")

    if predictions.dim() == 2:  # [H, W]
        predictions = predictions.unsqueeze(0).unsqueeze(0)
        targets = targets.unsqueeze(0).unsqueeze(0)
    elif predictions.dim() == 3:  # [C, H, W]
        predictions = predictions.unsqueeze(0)
        targets = targets.unsqueeze(0)

    # Compute edge maps
    edge_preds = compute_edge_map(predictions)
    edge_targets = compute_edge_map(targets)

    # Threshold the edge maps
    edge_preds = (edge_preds > edge_preds.mean()).float()
    edge_targets = (edge_targets > edge_targets.mean()).float()

    # Flatten the tensors
    edge_preds_flat = edge_preds.view(-1)
    edge_targets_flat = edge_targets.view(-1)

    # Ensure EPSILON is on the same device and data type as predictions
    epsilon = torch.tensor(EPSILON, device=predictions.device, dtype=predictions.dtype)

    # Calculate intersection and union
    intersection = (edge_preds_flat * edge_targets_flat).sum()
    dice = (2.0 * intersection + epsilon) / (edge_preds_flat.sum() + edge_targets_flat.sum() + epsilon)

    return dice.item()
