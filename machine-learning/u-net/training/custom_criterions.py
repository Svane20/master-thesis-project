import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """
        Calculate the Dice Loss

        Args:
            inputs (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels
            smooth (int): Smoothing factor to avoid division by zero

        Returns:
            torch.Tensor: Dice Loss
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined loss of Binary Cross Entropy (BCE) and Dice Loss
    """

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the BCE and Dice Loss

        Args:
            inputs (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Combined BCE and Dice Loss
        """
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return bce_loss + dice_loss
