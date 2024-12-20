import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from utils.edge_detection import compute_edge_map


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, laplace_smooth: int = 1) -> torch.Tensor:
    """
    Compute the DICE loss between predicted and ground truth masks.

    Args:
        inputs (torch.Tensor): Predicted masks.
        targets (torch.Tensor): Ground truth masks.
        laplace_smooth (int): Smoothing factor to avoid division by zero, defaults to 1.

    Returns:
        torch.Tensor: DICE loss tensor.
    """
    assert inputs.dim() == 4 and targets.dim() == 4, "Expected 4D tensors"

    # Apply sigmoid to predictions to get probabilities in the range [0, 1]
    inputs = torch.sigmoid(inputs)

    # Flatten tensors to 1D for easy calculation of overlap
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Calculate the intersection (common positive pixels) between inputs and targets
    intersection = (inputs * targets).sum()

    # Calculate Dice coefficient
    # Dice = (2 * |X ∩ Y|) / (|X| + |Y|), using smooth to prevent division by zero
    dice = (2.0 * intersection + laplace_smooth) / (inputs.sum() + targets.sum() + laplace_smooth)

    # Dice Loss (1 - Dice coefficient)
    # A lower Dice Loss indicates better overlap with the ground truth
    return 1 - dice


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25
) -> torch.Tensor:
    """
    Compute the Sigmoid Focal Loss between predicted and ground truth masks.

    Args:
        inputs (torch.Tensor): Predicted masks.
        targets (torch.Tensor): Ground truth masks.
        gamma (float): Focusing parameter for modulating loss, defaults to 2.0.
        alpha (float): Weighting factor for positive samples, defaults to 0.25.

    Returns:
        torch.Tensor: Sigmoid Focal Loss tensor.
    """
    assert inputs.dim() == 4 and targets.dim() == 4, "Expected 4D tensors"

    # Flatten tensors to 1D for easy calculation of overlap
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)

    # Calculate BCE loss
    bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none')

    # Compute probabilities for the focal modulation factor
    probabilities = torch.sigmoid(inputs_flat)
    p_t = probabilities * targets_flat + (1 - probabilities) * (1 - targets_flat)

    # Apply the focal factor
    loss = bce_loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets_flat + (1 - alpha) * (1 - targets_flat)
        loss = alpha_t * loss

    return loss.mean()


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 loss (mean absolute error) between predicted and ground truth alpha mattes.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.
    """
    return torch.mean(torch.abs(pred - gt))


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute a gradient loss to encourage matching edges between predicted and ground truth alpha.

    We approximate image gradients using a Sobel filter or simple finite differences.
    Here, simple finite differences to compute horizontal and vertical gradients.
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


def matting_loss(pred: torch.Tensor, gt: torch.Tensor, lambda_factor: float = 1.0):
    # Compute L1 loss
    loss_l1 = l1_loss(pred, gt)

    # Compute gradient loss
    loss_grad = gradient_loss(pred, gt)

    # Combine losses
    loss = loss_l1 + lambda_factor * loss_grad

    return loss


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
            'core_loss': loss,
            'l1_loss': loss_l1,
            'gradient_loss': loss_grad
        }


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.

    The Dice Loss is designed to measure the overlap between the predicted mask
    and the ground truth mask, especially in cases where the classes are imbalanced.
    This loss is effective for tasks like binary segmentation where the goal is
    to maximize the overlap between the predicted and actual regions of interest.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """
        Calculate the Dice Loss between predicted and ground truth masks.

        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth binary masks.
            smooth (int): A smoothing factor to avoid division by zero, defaults to 1.

        Returns:
            torch.Tensor: Dice Loss, a value between 0 and 1, where lower values indicate better overlap.
        """
        # Apply sigmoid to predictions to get probabilities in the range [0, 1]
        inputs = torch.sigmoid(inputs)

        # Flatten tensors to 1D for easy calculation of overlap
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate the intersection (common positive pixels) between inputs and targets
        intersection = (inputs * targets).sum()

        # Calculate Dice coefficient
        # Dice = (2 * |X ∩ Y|) / (|X| + |Y|), using smooth to prevent division by zero
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Dice Loss (1 - Dice coefficient)
        # A lower Dice Loss indicates better overlap with the ground truth
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined loss of Binary Cross Entropy (BCE) and Dice Loss.

    This loss function combines the strengths of both BCE and Dice Loss.
    BCE is effective for pixel-wise classification, while Dice Loss helps improve overlap between the predicted mask and ground truth.

    Args:
        bce_weight (float): Weight for the BCE loss term. Default is 0.5.
        dice_weight (float): Weight for the Dice loss term. Default is 0.5.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()

        # Initialize BCE with logits loss
        self.bce = nn.BCEWithLogitsLoss()

        # Initialize Dice loss
        self.dice = DiceLoss()

        # Weights for BCE and Dice Loss
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined BCE and Dice Loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: The combined loss, weighted by bce_weight and dice_weight.
        """
        # Calculate BCE loss
        bce_loss = self.bce(inputs, targets)

        # Calculate Dice loss
        dice_loss = self.dice(inputs, targets)

        # Combine the BCE and Dice losses using their respective weights
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


class EdgeWeightedBCEDiceLoss(nn.Module):
    """
    Combined loss of Binary Cross Entropy (BCE) and Dice Loss with edge weights.

    This loss function combines the strengths of both BCE and Dice Loss.
    BCE is effective for pixel-wise classification, while Dice Loss helps improve overlap between the predicted mask and ground truth.
    Additionally, this loss function applies edge weights in the ground truth mask to the BCE to improve edge detection,
    making the model focus more on accurately segmenting edge regions while still considering the entire mask.

    Args:
        edge_weight (float): Additional weight applied to edge regions in the BCE loss term. Default is 5.0.
        edge_loss_weight (float): Weight applied to the edge loss term. Default is 1.0.
    """

    def __init__(self, edge_weight: float = 5.0, edge_loss_weight: float = 1.0):
        super().__init__()

        # Initialize BCE with logits loss without reduction (we'll apply custom weighting)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # Standard Dice loss to ensure segmentation accuracy across the whole mask
        self.dice = DiceLoss()

        # Weight multiplier applied to BCE loss at edge pixels
        self.edge_weight = edge_weight

        # Weight applied to the edge loss term
        self.edge_loss_weight = edge_loss_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined BCE and Dice Loss with edge weights.

        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: The combined loss, with edge-weighted BCE and Dice Loss.
        """
        # Ensure inputs and targets are 4D or 3D tensors
        if inputs.dim() == 4:
            pass
        elif inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)
        else:
            raise ValueError('Expected inputs to be 3D or 4D tensor')

        # Ensure targets are float for computation
        targets = targets.float()

        # Compute edge map from ground truth mask
        edges_gt = compute_edge_map(targets)

        # Create edge weight mask for BCE loss
        edge_weight_mask = 1 + self.edge_weight * edges_gt

        # Calculate BCE loss with edge weighting
        bce_loss = self.bce(inputs, targets)

        # Apply edge weighting to the BCE loss
        weighted_bce_loss = (bce_loss * edge_weight_mask).mean()

        # Calculate Dice loss
        dice_loss = self.dice(inputs, targets)

        # Apply sigmoid to convert logits to probabilities
        inputs_prob = torch.sigmoid(inputs)

        # Compute edge map from predicted mask
        edges_pred = compute_edge_map(inputs_prob)

        # Calculate edge loss using L1 loss
        edge_loss = torch.nn.functional.l1_loss(edges_pred, edges_gt)

        # Combine the weighted BCE, Dice, and edge loss
        total_loss = weighted_bce_loss + dice_loss + self.edge_loss_weight * edge_loss

        return total_loss
