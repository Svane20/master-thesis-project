import torch
import torch.nn as nn

from utils.edge_detection import compute_edge_map


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
