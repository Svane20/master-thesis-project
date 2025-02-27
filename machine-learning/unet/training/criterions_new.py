import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

CORE_LOSS_KEY = 'core_loss'

def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient loss between predicted and ground truth alpha mattes.
    Approximates gradients using fixed Sobel filters.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: Gradient loss.
    """
    # Define Sobel filters for x and y gradients.
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    # Compute gradients for prediction and ground truth.
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_gt_x   = F.conv2d(gt, sobel_x, padding=1)
    grad_gt_y   = F.conv2d(gt, sobel_y, padding=1)

    # Compute the L1 difference of gradients.
    loss_x = torch.abs(grad_pred_x - grad_gt_x).mean()
    loss_y = torch.abs(grad_pred_y - grad_gt_y).mean()

    return loss_x + loss_y

class MattingLossV2(nn.Module):
    """
    Combined loss function for alpha matting tasks using only reconstruction (L1)
    and gradient losses.
    """

    def __init__(self, weight_dict: Dict[str, float], device: torch.device, dtype: torch.dtype) -> None:
        """
        Args:
            weight_dict (Dict[str, float]): Dictionary with keys "reconstruction" and "gradient"
                                            indicating their weights.
            device (torch.device): Device for loss computations.
            dtype (torch.dtype): Data type for loss computations.
        """
        super().__init__()

        # Optionally normalize weights.
        total_weight = sum(weight_dict.values())
        self.weight_dict = {k: v / total_weight for k, v in weight_dict.items()}

        for key in ["reconstruction", "gradient"]:
            if key not in self.weight_dict:
                raise ValueError(f"Loss weight for '{key}' must be provided.")

        # Register buffers for monitoring individual loss components.
        self.register_buffer("reconstruction_loss", torch.tensor(0.0, dtype=dtype, device=device))
        self.register_buffer("gradient_loss", torch.tensor(0.0, dtype=dtype, device=device))
        self.register_buffer(CORE_LOSS_KEY, torch.tensor(0.0, dtype=dtype, device=device))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the loss components.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing individual loss components and the total loss.
        """
        # Reset buffers.
        self.reconstruction_loss.zero_()
        self.gradient_loss.zero_()
        self.core_loss.zero_()

        losses = self._forward(pred, gt)

        # Update registered buffers.
        self.reconstruction_loss = losses["reconstruction"].to(dtype=self.reconstruction_loss.dtype,
                                                                device=self.reconstruction_loss.device)
        self.gradient_loss = losses["gradient"].to(dtype=self.gradient_loss.dtype,
                                                    device=self.gradient_loss.device)
        self.core_loss = losses[CORE_LOSS_KEY].to(dtype=self.core_loss.dtype,
                                                  device=self.core_loss.device)

        return losses

    def reduce_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reduce the losses by taking the weighted sum of the loss components.

        Args:
            losses (Dict[str, torch.Tensor]): Dictionary of loss components.

        Returns:
            torch.Tensor: Combined (core) loss.
        """
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"Loss key '{loss_key}' not computed.")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        return reduced_loss

    def _forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the individual loss components.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys "reconstruction", "gradient", and "core_loss".
        """
        losses = {}
        # L1 (reconstruction) loss.
        losses["reconstruction"] = F.l1_loss(pred, gt, reduction="mean")
        # Gradient loss.
        losses["gradient"] = gradient_loss(pred, gt)
        # Total loss (weighted sum).
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

# Example usage:
if __name__ == "__main__":
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Simulate predictions and ground truth with shape [B, 1, H, W].
    pred = torch.rand((4, 1, 256, 256), dtype=dtype, device=device, requires_grad=True)
    gt   = torch.rand((4, 1, 256, 256), dtype=dtype, device=device)
    # Define loss weights: e.g. reconstruction loss weight 1.0 and gradient loss weight λ.
    config = {"reconstruction": 1.0, "gradient": 0.5}

    loss_fn = MattingLossV2(weight_dict=config, device=device, dtype=dtype)
    losses = loss_fn(pred, gt)

    print(f"Reconstruction Loss: {losses['reconstruction']:.4f}")
    print(f"Gradient Loss: {losses['gradient']:.4f}")
    print(f"Total (Core) Loss: {losses[CORE_LOSS_KEY]:.4f}")
