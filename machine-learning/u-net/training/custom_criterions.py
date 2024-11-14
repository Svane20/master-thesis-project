import torch
import torch.nn as nn
from scipy.ndimage import sobel
import numpy as np


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
        # Step 1: Apply sigmoid to predictions to get probabilities in the range [0, 1]
        inputs = torch.sigmoid(inputs)

        # Flatten tensors to 1D for easy calculation of overlap
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Step 2: Calculate the intersection (common positive pixels) between inputs and targets
        intersection = (inputs * targets).sum()

        # Step 3: Calculate Dice coefficient
        # Dice = (2 * |X ∩ Y|) / (|X| + |Y|), using smooth to prevent division by zero
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Step 4: Return Dice Loss (1 - Dice coefficient)
        # A lower Dice Loss indicates better overlap with the ground truth
        return 1 - dice


class EdgeWeightedBCEDiceLoss(nn.Module):
    """
    Combined loss of Binary Cross Entropy (BCE) and Dice Loss with edge weights.

    This loss function combines the strengths of both BCE and Dice Loss.
    BCE is effective for pixel-wise classification, while Dice Loss helps improve overlap between the predicted mask and ground truth.
    Additionally, this loss function applies edge weights in the ground truth mask to the BCE to improve edge detection,
    making the model focus more on accurately segmenting edge regions while still considering the entire mask.

    Args:
        edge_weight (float): Additional weight applied to edge regions in the BCE loss term.
    """

    def __init__(self, edge_weight=5):
        super().__init__()

        # Initialize BCE with logits loss without reduction (we'll apply custom weighting)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # Standard Dice loss to ensure segmentation accuracy across the whole mask
        self.dice = DiceLoss()

        # Weight multiplier applied to BCE loss at edge pixels
        self.edge_weight = edge_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined BCE and Dice Loss with edge weights.

        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: The combined loss, with edge-weighted BCE and Dice Loss.
        """
        # Step 1: Calculate edge map for the target mask
        # Apply Sobel filter to detect edges in the ground truth mask
        targets_np = targets.cpu().numpy()

        # Use Sobel filter to create an edge map, marking edges in the ground truth mask
        edges_np = np.array([sobel(target, axis=0) + sobel(target, axis=1) for target in targets_np])
        edges = torch.tensor(edges_np, device=targets.device)  # Convert edges back to torch tensor
        edges = (edges > 0).float()  # Convert edge map to binary mask, where edges are marked as 1

        # Step 2: Apply edge weighting to BCE loss
        # Calculate BCE loss at each pixel and apply extra weight to edge pixels
        # The term (1 + self.edge_weight * edges) boosts loss at edges
        weighted_bce_loss = (self.bce(inputs, targets) * (1 + self.edge_weight * edges)).mean()

        # Step 3: Calculate Dice loss across the entire mask
        # Dice loss ensures overall segmentation accuracy, balancing the focus on edges
        dice_loss = self.dice(inputs, targets)

        # Step 4: Combine weighted BCE and Dice loss
        # Weighted BCE encourages better edge accuracy, while Dice maintains overall mask fidelity
        return weighted_bce_loss + dice_loss


class BCEDiceLoss(nn.Module):
    """
    Combined loss of Binary Cross Entropy (BCE) and Dice Loss for binary segmentation tasks.

    This loss function combines the strengths of both BCE and Dice Loss. BCE is effective for
    pixel-wise classification, while Dice Loss helps improve overlap between the predicted
    mask and ground truth. Together, they provide a balanced loss that encourages the model
    to accurately classify individual pixels and maximize overlap with the ground truth mask.
    """

    def __init__(self):
        super().__init__()

        # Binary Cross Entropy with Logits Loss for pixel-wise classification
        self.bce = nn.BCEWithLogitsLoss()

        # Dice Loss to measure overlap and improve segmentation accuracy
        self.dice = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined BCE and Dice Loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: The combined BCE and Dice Loss, where a lower value indicates
                          better alignment between predictions and ground truth.
        """
        # Step 1: Compute Binary Cross Entropy Loss
        # BCE Loss is effective at classifying individual pixels correctly
        bce_loss = self.bce(inputs, targets)

        # Step 2: Compute Dice Loss
        # This ensures higher overlap between the predicted and actual masks
        dice_loss = self.dice(inputs, targets)

        # Step 3: Combine BCE and Dice Loss
        # Adding BCE and Dice Loss balances pixel-wise accuracy with overall segmentation overlap
        return bce_loss + dice_loss
