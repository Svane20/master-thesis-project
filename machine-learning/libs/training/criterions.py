import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from .utils.criterion_utils import gauss_kernel, laplacian_pyramid, compute_boundary_map

CORE_LOSS_KEY = 'core_loss'


def laplacian_loss(predictions: torch.Tensor, targets: torch.Tensor, max_levels: int = 5) -> torch.Tensor:
    """
    Compute the Laplacian loss between predicted and ground truth alpha mattes.

    Args:
        predictions (torch.Tensor): Predicted alpha matte.
        targets (torch.Tensor): Ground truth alpha matte.
        max_levels (int): Number of levels in the Laplacian pyramid. Default is 5.

    Returns:
        torch.Tensor: Laplacian loss.
    """
    kernel = gauss_kernel(device=predictions.device, dtype=predictions.dtype)

    # Create Laplacian pyramids
    pred_pyramid = laplacian_pyramid(predictions, kernel, max_levels)
    gt_pyramid = laplacian_pyramid(targets, kernel, max_levels)

    # Compute Laplacian loss
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], gt_pyramid[level])

    # Normalize loss
    return loss / max_levels


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient loss between predicted and ground truth alpha mattes.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: Gradient loss.
    """
    # Sobel filters for computing gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    # Compute gradients for prediction and ground truth
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_gt_x = F.conv2d(gt, sobel_x, padding=1)
    grad_gt_y = F.conv2d(gt, sobel_y, padding=1)

    # Compute gradient differences
    grad_diff_x = torch.abs(grad_pred_x - grad_gt_x)
    grad_diff_y = torch.abs(grad_pred_y - grad_gt_y)

    # Combine gradient losses
    grad_loss = torch.mean(grad_diff_x + grad_diff_y)

    return grad_loss


def boundary_aware_loss(pred: torch.Tensor, gt: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
    """
    Compute boundary-aware loss, emphasizing boundaries of alpha mattes.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.
        boundary_map (torch.Tensor): Precomputed boundary map indicating the boundaries.

    Returns:
        torch.Tensor: Boundary-aware loss.
    """
    l1_loss = F.l1_loss(pred, gt, reduction="none")

    # Apply boundary map to emphasize boundary regions (0, 0.5, 1)
    semi_transparent_mask = (gt > 0.0) & (gt < 1.0)

    # Weighted loss
    weighted_loss = l1_loss * (boundary_map + semi_transparent_mask.float())

    return torch.mean(weighted_loss)


class MattingLoss(nn.Module):
    """
    Combined loss function for alpha matting tasks.
    """

    def __init__(self, weight_dict: Dict[str, float], device: torch.device, dtype: torch.dtype) -> None:
        """
        Args:
            weight_dict (Dict[str, float]): Dictionary containing weights for each loss component.
            device (torch.device): Device to run the loss on.
            dtype (torch.dtype): Data type for the loss.
        """
        super().__init__()

        # Normalize weights
        total_weight = sum(weight_dict.values())
        self.weight_dict = {k: v / total_weight for k, v in weight_dict.items()}

        for key in ["reconstruction", "laplacian", "gradient", "boundary"]:
            assert key in self.weight_dict, f"{key} loss weight must be provided."

        # Initialize buffers to store the losses
        self.register_buffer(name="reconstruction_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="laplacian_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="gradient_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="boundary_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name=CORE_LOSS_KEY, tensor=torch.tensor(data=0.0, dtype=dtype, device=device))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the loss.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components.
        """
        # Reset buffers
        self.reconstruction_loss.zero_()
        self.laplacian_loss.zero_()
        self.gradient_loss.zero_()
        self.boundary_loss.zero_()
        self.core_loss.zero_()

        # Compute losses
        losses = self._forward(pred, gt)

        # Update buffers
        self.reconstruction_loss = losses["reconstruction"].to(dtype=self.reconstruction_loss.dtype,
                                                               device=self.reconstruction_loss.device)
        self.laplacian_loss = losses["laplacian"].to(dtype=self.laplacian_loss.dtype, device=self.laplacian_loss.device)
        self.gradient_loss = losses["gradient"].to(dtype=self.gradient_loss.dtype, device=self.gradient_loss.device)
        self.boundary_loss = losses["boundary"].to(dtype=self.boundary_loss.dtype, device=self.boundary_loss.device)
        self.core_loss = losses[CORE_LOSS_KEY].to(dtype=self.core_loss.dtype, device=self.core_loss.device)

        return losses

    def reduce_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reduce the loss by summing the weighted loss components.

        Args:
            losses (Dict[str, torch.Tensor]): Dictionary of loss components.

        Returns:
            torch.Tensor: Reduced loss.
        """
        reduced_loss = 0.0

        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")

            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss

    def _forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the loss.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        losses = {"reconstruction": 0, "laplacian": 0, "gradient": 0, "boundary": 0}

        # Update losses
        self._update_losses(losses, pred, gt)

        # Reduce losses
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)

        return losses

    def _update_losses(
            self,
            losses: Dict[str, torch.Tensor],
            pred: torch.Tensor,
            gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update the loss components.

        Args:
            losses (Dict[str, torch.Tensor]): Dictionary of loss components.
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Updated dictionary of loss components.
        """
        # Reconstruction loss (L1 loss)
        losses["reconstruction"] = F.l1_loss(pred, gt, reduction="mean")

        # Laplacian loss
        losses["laplacian"] = laplacian_loss(pred, gt) / pred.size(0)

        # Compute gradient loss
        losses["gradient"] = gradient_loss(pred, gt)

        # Compute boundary-aware loss
        boundary_map = compute_boundary_map(gt)
        losses["boundary"] = boundary_aware_loss(pred, gt, boundary_map)

        return losses


if __name__ == "__main__":
    # Generate some random predictions and ground truth
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = torch.rand((4, 1, 256, 256), dtype=dtype, requires_grad=True).to(device)
    targets = torch.rand((4, 1, 256, 256), dtype=dtype).to(device)
    config = {"reconstruction": 1.0, "laplacian": 1.0, "gradient": 0.5, "boundary": 0.5}

    with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=dtype):
        # Compute the loss
        loss_fn = MattingLoss(weight_dict=config, device=device, dtype=dtype)
        losses = loss_fn(predictions, targets)

        # Print the losses
        print(f"Reconstruction Loss: {losses['reconstruction']:.4f}")
        print(f"Laplacian Loss: {losses['laplacian']:.4f}")
        print(f"Gradient Loss: {losses['gradient']:.4f}")
        print(f"Boundary Loss: {losses['boundary']:.4f}")
        print(f"Core Loss: {losses[CORE_LOSS_KEY]:.4f}\n")

        # Print state_dict
        print(f"Loss function state dict: {loss_fn.state_dict()}")

    # Save state_dict
    torch.save(loss_fn.state_dict(), "matting_loss.pth")

    # Step 3: Create a new loss function instance
    new_loss_fn = MattingLoss(weight_dict=config, device=device, dtype=dtype)

    # Step 4: Load the saved state dict into the new instance
    state_dict = torch.load("matting_loss.pth", weights_only=True)
    new_loss_fn.load_state_dict(state_dict)

    print("\nLoaded State Dict:")
    for key, value in new_loss_fn.state_dict().items():
        print(f"{key}: {value:.4f}")

    # Step 5: Verify that buffers match
    assert torch.allclose(loss_fn.reconstruction_loss, new_loss_fn.reconstruction_loss), "Reconstruction loss mismatch"
    assert torch.allclose(loss_fn.laplacian_loss, new_loss_fn.laplacian_loss), "Laplacian loss mismatch"
    assert torch.allclose(loss_fn.gradient_loss, new_loss_fn.gradient_loss), "Gradient loss mismatch"
    assert torch.allclose(loss_fn.boundary_loss, new_loss_fn.boundary_loss), "Boundary loss mismatch"
    assert torch.allclose(loss_fn.core_loss, new_loss_fn.core_loss), "Core loss mismatch"

    print("\nState dict successfully loaded and verified!")
