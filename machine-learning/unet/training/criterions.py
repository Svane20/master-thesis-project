import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 loss (mean absolute error) between predicted and ground truth alpha mattes.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: L1 loss.
    """
    return torch.mean(torch.abs(pred - gt))


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute a gradient loss to encourage matching edges between predicted and ground truth alpha.

    We approximate image gradients using a Sobel filter or simple finite differences.
    Here, simple finite differences to compute horizontal and vertical gradients.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: Gradient loss.
    """
    # Horizontal and vertical gradient kernels for finite differences
    kernel_x = torch.tensor(
        data=[[-1, 1], [0, 0]],
        dtype=pred.dtype,
        device=pred.device
    ).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor(
        data=[[-1, 0], [1, 0]],
        dtype=pred.dtype,
        device=pred.device
    ).unsqueeze(0).unsqueeze(0)

    # Ensure shape: [B, 1, H, W]
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    # Compute gradients via convolution
    pred_gx = F.conv2d(pred, kernel_x, padding=0)
    pred_gy = F.conv2d(pred, kernel_y, padding=0)
    gt_gx = F.conv2d(gt, kernel_x, padding=0)
    gt_gy = F.conv2d(gt, kernel_y, padding=0)

    # Calculate L1 difference in gradients
    loss_x = torch.mean(torch.abs(pred_gx - gt_gx))
    loss_y = torch.mean(torch.abs(pred_gy - gt_gy))

    return (loss_x + loss_y) / 2.0


def laplacian_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute a loss that emphasizes fine details using a Laplacian filter.
    This encourages sharper edges and better fine-grained detail alignment.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: Laplacian loss.
    """
    # Laplacian kernel
    lap_kernel = torch.tensor(
        data=[[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=pred.dtype,
        device=pred.device
    ).unsqueeze(0).unsqueeze(0)

    # Ensure shape: [B, 1, H, W]
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    pred_lap = F.conv2d(pred, lap_kernel, padding=1)
    gt_lap = F.conv2d(gt, lap_kernel, padding=1)

    return torch.mean(torch.abs(pred_lap - gt_lap))


class CombinedMattingLoss(nn.Module):
    """
    Combined loss function for alpha matting tasks.
    """

    def __init__(self, lambda_factor: float = 1.0):
        """
        Args:
            lambda_factor (float): Weight factor for the gradient loss term. Default is 1.0.
        """
        super().__init__()

        self.lambda_factor = lambda_factor

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss for alpha matting tasks.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the total loss, L1 loss, and gradient loss.
        """
        # Compute L1 loss
        loss_l1 = l1_loss(pred, gt)

        # Compute gradient loss
        loss_grad = gradient_loss(pred, gt)

        # Combine losses
        loss = loss_l1 + self.lambda_factor * loss_grad

        return {
            'loss': loss,
            'l1_loss': loss_l1,
            'gradient_loss': loss_grad
        }
