import torch
import torch.nn.functional as F

from typing import Dict, Tuple
from skimage.measure import label
import numpy as np


def get_grad_filter(device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Get the gradient filter for computing the gradient loss.

    Args:
        device (torch.device): Device to run the filter on.
        dtype (torch.dtype): Data type of the filter. Default is torch.float16.

    Returns:
        torch.Tensor: Gradient filter.
    """
    y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    grad_filter = torch.tensor([y, x], dtype=dtype, device=device)
    grad_filter = grad_filter.unsqueeze(1)

    return grad_filter


def compute_training_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    Compute training metrics for a batch of alpha matte predictions.

    Args:
        pred (torch.Tensor): Predicted alpha mattes (B, C, H, W).
        gt (torch.Tensor): Ground truth alpha mattes (B, C, H, W).

    Returns:
        Dict[str, float]: Dictionary of averaged metrics for the batch.
    """
    # Resize tensors if necessary
    pred, gt = _resize_tensors(pred, gt)

    # Calculate metrics
    mae = F.l1_loss(pred, gt, reduction="mean").item()
    mse = F.mse_loss(pred, gt, reduction="mean").item()

    return {"mae": mae, "mse": mse}


def compute_evaluation_metrics(
        preds: torch.Tensor,
        gts: torch.Tensor,
        grad_filter: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a batch of alpha matte predictions.

    Args:
        preds (torch.Tensor): Predicted alpha mattes (B, C, H, W).
        gts (torch.Tensor): Ground truth alpha mattes (B, C, H, W).
        grad_filter (torch.Tensor): Gradient filter.

    Returns:
        Dict[str, float]: Dictionary of averaged metrics for the batch.
    """
    # Resize tensors if necessary
    preds, gts = _resize_tensors(preds, gts)

    l1_list = []
    l2_list = []
    sad_list = []
    grad_list = []
    conn_list = []

    for pred, gt in zip(preds, gts):
        l1_dist = F.l1_loss(pred, gt)
        l2_dist = F.mse_loss(pred, gt)
        sad = _compute_sad(pred, gt)
        grad = _compute_grad(pred, gt, grad_filter)
        conn = _compute_connectivity(pred, gt)

        l1_list.append(l1_dist)
        l2_list.append(l2_dist)
        sad_list.append(sad)
        grad_list.append(grad)
        conn_list.append(conn)

    l1_dist = torch.stack(l1_list, dim=0)
    l2_dist = torch.stack(l2_list, dim=0)
    sad_error = torch.stack(sad_list, dim=0)
    grad_error = torch.stack(grad_list, dim=0)
    conn_error = torch.stack(conn_list, dim=0)

    return {
        "mae": l1_dist.mean().item(),
        "mse": l2_dist.mean().item(),
        "sad": sad_error.mean().item(),
        "grad": grad_error.mean().item(),
        "conn": conn_error.mean().item()
    }


def _compute_sad(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Sum of Absolute Differences (SAD) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        target (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: SAD value.
    """
    error_map = torch.abs((pred - target))
    loss = torch.sum(error_map)

    return loss / 1000.0


def _compute_grad(
        preds: torch.Tensor,
        labels: torch.Tensor,
        grad_filter: torch.Tensor,
        epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculate gradient loss between predictions and targets.

    Args:
        preds (torch.Tensor): Predicted alpha matte.
        labels (torch.Tensor): Ground truth alpha matte.
        grad_filter (torch.Tensor): Gradient filter.
        epsilon (float): Small value to prevent division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: Gradient loss.
    """
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    grad_preds = F.conv2d(preds, weight=grad_filter, padding=1)
    grad_labels = F.conv2d(labels, weight=grad_filter, padding=1)
    grad_preds = torch.sqrt((grad_preds * grad_preds).sum(dim=1, keepdim=True) + epsilon)
    grad_labels = torch.sqrt(
        (grad_labels * grad_labels).sum(dim=1, keepdim=True) + 1e-8
    )

    return F.l1_loss(grad_preds, grad_labels)


def _compute_connectivity(pred: torch.Tensor, target: torch.Tensor, step: float = 0.1) -> torch.Tensor:
    thresh_steps = list(torch.arange(0, 1 + step, step))
    l_map = torch.ones_like(pred, dtype=torch.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).to(dtype=torch.int)
        target_alpha_thresh = (target >= thresh_steps[i]).to(dtype=torch.int)

        omega = torch.from_numpy(_get_largest_cc(pred_alpha_thresh * target_alpha_thresh)).to(pred.device,
                                                                                              dtype=torch.int)
        flag = ((l_map == -1) & (omega == 0)).to(dtype=torch.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).to(dtype=torch.int)
    target_phi = 1 - target_d * (target_d >= 0.15).to(dtype=torch.int)
    loss = torch.sum(torch.abs(pred_phi - target_phi))

    return loss / 1000.0


def _get_largest_cc(segmentation: torch.Tensor) -> np.ndarray:
    """
    Get the largest connected component in the segmentation

    Args:
        segmentation (torch.Tensor): Segmentation mask.

    Returns:
        torch.Tensor: Largest connected component.
    """
    segmentation = segmentation.detach().cpu().numpy()
    labels = label(segmentation, connectivity=1)
    if labels.max() == 0:
        return np.zeros_like(segmentation, dtype=bool)

    return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1  # Ignore background label


def _resize_tensors(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha_pred.dim() == 2:
        alpha_pred = alpha_pred.unsqueeze(0).unsqueeze(0)
        alpha_gt = alpha_gt.unsqueeze(0).unsqueeze(0)
    elif alpha_pred.dim() == 3:
        alpha_pred = alpha_pred.unsqueeze(0)
        alpha_gt = alpha_gt.unsqueeze(0)

    return alpha_pred, alpha_gt
