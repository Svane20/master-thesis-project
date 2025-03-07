import torch
import torch.nn.functional as F

from typing import Dict, Tuple
import numpy as np
import cv2


def compute_training_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    Compute training metrics for a batch of alpha matte predictions.

    Args:
        pred (torch.Tensor): Predicted alpha mattes (B, C, H, W).
        gt (torch.Tensor): Ground truth alpha mattes (B, C, H, W).

    Returns:
        Dict[str, float]: Dictionary of averaged metrics for the batch.
    """
    # Initialize accumulators for metrics
    batch_metrics = {"mse": 0.0, "mae": 0}
    batch_size = pred.size(0)

    # Compute metrics for each sample in the batch
    for i in range(batch_size):
        sample_metrics = compute_metrics(pred[i:i + 1], gt[i:i + 1])  # Process one sample at a time

        for key in batch_metrics:
            batch_metrics[key] += sample_metrics[key]

    # Average metrics over the batch
    for key in batch_metrics:
        batch_metrics[key] /= batch_size

    return batch_metrics


def compute_evaluation_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics for a batch of alpha matte predictions.

    Args:
        pred (torch.Tensor): Predicted alpha mattes (B, C, H, W).
        gt (torch.Tensor): Ground truth alpha mattes (B, C, H, W).

    Returns:
        Dict[str, float]: Dictionary of averaged metrics for the batch.
    """
    # Initialize accumulators for metrics
    batch_metrics = {"sad": 0.0, "mse": 0.0, "mae": 0, "grad": 0.0, "conn": 0.0}
    batch_size = pred.size(0)

    # Compute metrics for each sample in the batch
    for i in range(batch_size):
        sample_metrics = compute_metrics(pred[i:i + 1], gt[i:i + 1], is_eval=True)  # Process one sample at a time

        for key in batch_metrics:
            batch_metrics[key] += sample_metrics[key]

    # Average metrics over the batch
    for key in batch_metrics:
        batch_metrics[key] /= batch_size

    return batch_metrics


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, is_eval: bool = False) -> Dict[str, float]:
    """
    Compute evaluation metrics for alpha matte prediction.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.
        is_eval (bool): Whether to compute evaluation metrics. Default is False.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    metrics = {
        "mse": compute_MSE(pred, gt),
        "mae": compute_MAE(pred, gt),
    }

    if is_eval:
        metrics["sad"] = compute_SAD(pred, gt),
        metrics["conn"] = compute_CONN(pred, gt, is_batch=False)
        metrics["grad"] = compute_GRAD(pred, gt)

    return metrics


def compute_MSE(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error (MSE) between predictions and targets.

    Args:
        alpha_pred (torch.Tensor): Predicted alpha matte.
        alpha_gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: MSE value.
    """
    alpha_pred, alpha_gt = _resize_tensors(alpha_pred, alpha_gt)

    mse = torch.mean((alpha_pred - alpha_gt) ** 2, dim=[1, 2, 3])
    return mse if mse.numel() > 1 else mse.item()


def compute_MAE(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error (MAE) between predictions and targets.

    Args:
        alpha_pred (torch.Tensor): Predicted alpha matte.
        alpha_gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: MAE value.
    """
    alpha_pred, alpha_gt = _resize_tensors(alpha_pred, alpha_gt)

    mae = torch.mean(torch.abs(alpha_pred - alpha_gt), dim=[1, 2, 3])
    return mae if mae.numel() > 1 else mae.item()


def compute_SAD(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor) -> float:
    """
    Calculate Sum of Absolute Differences (SAD) between predictions and targets.

    Args:
        alpha_pred (torch.Tensor): Predicted alpha matte.
        alpha_gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: SAD value.
    """
    alpha_pred, alpha_gt = _resize_tensors(alpha_pred, alpha_gt)

    sad = torch.sum(torch.abs(alpha_pred - alpha_gt), dim=[1, 2, 3])
    return sad if sad.numel() > 1 else sad.item()


def compute_GRAD(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor) -> float:
    """
    Calculate Gradient Error (GRAD) between predictions and targets.

    Args:
        alpha_pred (torch.Tensor): Predicted alpha matte.
        alpha_gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: GRAD value.
    """
    alpha_pred, alpha_gt = _resize_tensors(alpha_pred, alpha_gt)

    grad_pred_x, grad_pred_y = _compute_gradients(alpha_pred)
    grad_gt_x, grad_gt_y = _compute_gradients(alpha_gt)

    # Compute gradient magnitude difference
    grad_diff = torch.sqrt((grad_pred_x - grad_gt_x) ** 2 + (grad_pred_y - grad_gt_y) ** 2)

    # Sum over spatial dimensions (and channel) for each image in the batch
    grad_error = torch.sum(grad_diff, dim=[1, 2, 3])

    return grad_error if grad_error.numel() > 1 else grad_error.item()


def compute_CONN(
        alpha_pred: torch.Tensor,
        alpha_gt: torch.Tensor,
        is_batch: bool = False,
        **kwargs
):
    """
    Calculate Connectivity Error (CONN) between predictions and targets.

    Args:
        alpha_pred (torch.Tensor): Predicted alpha matte.
        alpha_gt (torch.Tensor): Ground truth alpha matte.
        is_batch (bool, optional): Whether the inputs are batched. Defaults to False.


    Returns:
        float: CONN value.
    """
    if is_batch:
        errors = []
        B = alpha_pred.size(0)

        for i in range(B):
            error = _compute_conn_single(alpha_pred[i, 0], alpha_gt[i, 0], **kwargs)
            errors.append(error)

        return torch.tensor(errors)
    else:
        return _compute_conn_single(alpha_pred, alpha_gt, **kwargs)


def _compute_conn_single(
        alpha_pred: torch.Tensor,
        alpha_gt: torch.Tensor,
        step: float = 0.1,
        eps: float = 1e-6
) -> float:
    """
    Calculate Connectivity Error (CONN) between predictions and targets.

    Args:
        alpha_pred (torch.Tensor): Predicted alpha matte.
        alpha_gt (torch.Tensor): Ground truth alpha matte.
        step (float, optional): Step size. Defaults to 0.1.
        eps (float, optional): Epsilon. Defaults to 1e-6.

    Returns:
        float: CONN value.
    """
    if isinstance(alpha_pred, torch.Tensor):
        alpha_pred = alpha_pred.detach().cpu().numpy().squeeze()
        alpha_gt = alpha_gt.detach().cpu().numpy().squeeze()

    # Absolute difference map
    abs_diff = np.abs(alpha_pred - alpha_gt)
    thresholds = np.arange(0, 1 + step, step)
    min_error = np.inf

    for t in thresholds:
        # Binarize based on threshold t
        pred_bin = (alpha_pred >= t).astype(np.uint8)
        gt_bin = (alpha_gt >= t).astype(np.uint8)

        # Connected components on ground truth
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)

        if num_labels > 1:
            # Skip label 0 (background) and find the largest component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_mask = (labels == largest_label).astype(np.uint8)

            # Compute error only within the largest connected region
            error = np.sum(abs_diff * largest_mask) / (np.sum(largest_mask) + eps)
            min_error = min(min_error, error)

    return min_error


def _compute_gradients(x):
    """
    Computes the x and y gradients using Sobel filters.
    x: tensor of shape [B, 1, H, W] (or single image with shape [1, H, W])

    Returns:
        tuple (grad_x, grad_y)
    """
    # Define Sobel kernels
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    # Apply convolution (padding=1 to preserve size)
    grad_x = F.conv2d(x, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(x, sobel_kernel_y, padding=1)

    return grad_x, grad_y


def _resize_tensors(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha_pred.dim() == 2:
        alpha_pred = alpha_pred.unsqueeze(0).unsqueeze(0)
        alpha_gt = alpha_gt.unsqueeze(0).unsqueeze(0)
    elif alpha_pred.dim() == 3:
        alpha_pred = alpha_pred.unsqueeze(0)
        alpha_gt = alpha_gt.unsqueeze(0)

    return alpha_pred, alpha_gt
