import torch
from scipy.ndimage import sobel
import numpy as np


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


def calculate_edge_IoU(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates Intersection over Union (IoU) between edge maps of predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.

    Returns:
        float: Intersection over Union (IoU) value.
    """
    # Convert predictions and targets to numpy arrays on CPU for Sobel processing
    targets_np = targets.cpu().numpy()
    preds_np = predictions.cpu().numpy()

    # Ensure each sample is 2D (H, W) by adding batch dimension if necessary
    if targets_np.ndim == 2:  # Single 2D image
        targets_np = np.expand_dims(targets_np, axis=0)  # Convert to [1, H, W]
    elif targets_np.ndim == 4:  # Batch with channel dimension [batch_size, 1, H, W]
        targets_np = np.squeeze(targets_np, axis=1)  # Remove channel dimension

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

    # Calculate Intersection over Union (IoU) for edges
    intersection = (edge_preds * edge_targets).sum()
    union = edge_preds.sum() + edge_targets.sum() - intersection
    edge_iou = (intersection + 1e-6) / (union + 1e-6)

    return edge_iou.item()
