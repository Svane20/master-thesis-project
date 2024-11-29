import torch
import torch.nn.functional as F


def iou(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute the Intersection over Union (IoU) score

    Args:
        preds (torch.Tensor): Predictions
        targets (torch.Tensor): Targets
        epsilon (float): Smoothing factor to avoid division by zero. Default: 1e-6

    Returns:
        float: IoU score
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.contiguous()
    targets = targets.contiguous()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def boundary_iou(preds: torch.Tensor, targets: torch.Tensor, boundary_width: int = 1) -> float:
    """
    Compute the Boundary IoU score using PyTorch operations to avoid CPU-GPU data transfers.

    Args:
        preds (torch.Tensor): Predictions (logits)
        targets (torch.Tensor): Targets (binary masks)
        boundary_width (int): Width of the boundary for dilation. Default: 1

    Returns:
        float: Boundary IoU score
    """
    # Apply sigmoid and threshold to get binary predictions
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    # Ensure targets are binary
    targets = (targets > 0.5).float()

    # Get boundary masks using Laplacian filter
    def get_boundary(mask):
        laplace_filter = torch.tensor([[[[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]]]], device=mask.device, dtype=mask.dtype)
        boundary = F.conv2d(mask, laplace_filter, padding=1)
        boundary = boundary.abs()
        boundary = torch.clamp(boundary, 0, 1)
        return boundary

    preds_boundary = get_boundary(preds)
    targets_boundary = get_boundary(targets)

    # Dilate boundaries using max pooling
    def binary_dilation(mask, kernel_size=3, iterations=1):
        pad = kernel_size // 2
        for _ in range(iterations):
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
        return mask

    preds_dilated = binary_dilation(preds_boundary, kernel_size=3, iterations=boundary_width)
    targets_dilated = binary_dilation(targets_boundary, kernel_size=3, iterations=boundary_width)

    # Compute intersection and union
    intersection = torch.sum(preds_dilated * targets_dilated)
    union = torch.sum((preds_dilated + targets_dilated) >= 1).float()

    boundary_iou = (intersection / union).item() if union > 0 else 0.0

    return boundary_iou