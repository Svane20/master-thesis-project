import torch
from scipy.ndimage import sobel
import numpy as np

EPSILON = 1e-6


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
    # Convert predictions and targets to numpy arrays on CPU for Sobel processing
    targets_np = targets.cpu().numpy()
    preds_np = predictions.cpu().numpy()

    # Check if we have a single image (2D) or a batch (3D/4D) and add batch dimension if needed
    if targets_np.ndim == 2:  # Single 2D image
        targets_np = np.expand_dims(targets_np, axis=0)  # Convert to [1, H, W]
    elif targets_np.ndim == 4:  # Batch with channel dimension [batch_size, 1, H, W]
        targets_np = np.squeeze(targets_np, axis=1)  # Remove channel dimension, resulting in [batch_size, H, W]

    if preds_np.ndim == 2:  # Single 2D image
        preds_np = np.expand_dims(preds_np, axis=0)  # Convert to [1, H, W]
    elif preds_np.ndim == 4:  # Batch with channel dimension [batch_size, 1, H, W]
        preds_np = np.squeeze(preds_np, axis=1)

    # Calculate edge maps using the Sobel filter for each sample in the batch
    edge_targets_list = [sobel(target, axis=0) + sobel(target, axis=1) for target in targets_np]
    edge_preds_list = [sobel(pred, axis=0) + sobel(pred, axis=1) for pred in preds_np]

    # Convert lists to numpy arrays, then to tensors
    edge_targets = torch.tensor(np.array(edge_targets_list), device=targets.device)
    edge_preds = torch.tensor(np.array(edge_preds_list), device=predictions.device)

    # Convert edge maps to binary (1 for edges, 0 otherwise)
    edge_targets = (edge_targets > 0).float()
    edge_preds = (edge_preds > 0).float()

    # Calculate Dice Coefficient for edges
    intersection = (edge_preds * edge_targets).sum()
    dice = (2.0 * intersection + EPSILON) / (edge_preds.sum() + edge_targets.sum() + EPSILON)

    return dice.item()
