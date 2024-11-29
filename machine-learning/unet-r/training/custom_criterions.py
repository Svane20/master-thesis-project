import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        loss = 1 - dice

        return loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_boundary = self.get_boundary(preds)
        targets_boundary = self.get_boundary(targets)

        # Compute loss using BCEWithLogitsLoss
        return self.bce_with_logits(preds_boundary, targets_boundary)

    @staticmethod
    def get_boundary(logits: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        laplace_filter = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=probs.device, dtype=probs.dtype)

        boundary = F.conv2d(probs, laplace_filter, padding=1)
        boundary = boundary.abs()

        return torch.clamp(boundary, 0, 1)
