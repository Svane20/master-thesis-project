import torch

from training.utils.edge_detection import compute_edge_map


def calculate_dice_score(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
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


def calculate_dice_edge_score(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Calculates Dice Coefficient between edge maps of predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.
        epsilon (float): Small value to avoid division by zero. Default is 1e-6.

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
    epsilon = torch.tensor(epsilon, device=predictions.device, dtype=predictions.dtype)

    # Calculate intersection and union
    intersection = (edge_preds_flat * edge_targets_flat).sum()
    dice = (2.0 * intersection + epsilon) / (edge_preds_flat.sum() + edge_targets_flat.sum() + epsilon)

    return dice.item()


def calculate_iou_score(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Calculates Intersection over Union (IoU) between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.
        epsilon (float): Small value to avoid division by zero. Default is 1e-6.

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
    epsilon = torch.tensor(epsilon, device=predictions.device, dtype=predictions.dtype)

    # Calculate IoU
    iou = (intersection + epsilon) / (union + epsilon)  # Add epsilon to prevent division by zero

    return iou.item()
