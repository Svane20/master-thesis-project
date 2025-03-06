from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

CORE_LOSS_KEY = 'core_loss'


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    L1 loss between predicted and ground truth alpha mattes.
    """
    loss = torch.abs(pred - gt).mean()
    return loss


def composition_loss(
        pred: torch.Tensor,
        gt: torch.Tensor,
        image: torch.Tensor = None,
        fg: torch.Tensor = None,
        bg: torch.Tensor = None
) -> torch.Tensor:
    """
    Composition loss ensures that the predicted alpha, when used to composite the image,
    reconstructs the ground truth composite.

    If fg and bg are not available, it uses the provided image (assumed to be the composite)
    and multiplies by alpha (i.e. black background for (1-alpha)).
    """
    if fg is not None and bg is not None:
        comp_pred = pred * fg + (1 - pred) * bg
        comp_gt = gt * fg + (1 - gt) * bg
    elif image is not None:
        comp_pred = pred * image
        comp_gt = gt * image
    else:
        raise ValueError("For composition loss, either (fg, bg) or image must be provided.")

    loss = torch.abs(comp_pred - comp_gt).mean()
    return loss


def laplacian_loss(pred: torch.Tensor, gt: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """
    Laplacian pyramid loss for multi-scale supervision on the alpha matte.
    """
    current_pred = pred
    current_gt = gt
    loss = 0.0

    # Build the pyramid for the specified number of scales.
    for level in range(1, levels):
        # Downsample current predictions and ground truth.
        pred_down = F.interpolate(current_pred, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt_down = F.interpolate(current_gt, scale_factor=0.5, mode='bilinear', align_corners=False)

        # Upsample back to current scale.
        pred_up = F.interpolate(pred_down, size=current_pred.shape[2:], mode='bilinear', align_corners=False)
        gt_up = F.interpolate(gt_down, size=current_gt.shape[2:], mode='bilinear', align_corners=False)

        # Compute the high-frequency band at this scale.
        lap_pred = current_pred - pred_up
        lap_gt = current_gt - gt_up

        band_loss = torch.abs(lap_pred - lap_gt).mean()
        loss += (2 ** (level - 1)) * band_loss

        # Prepare for next level.
        current_pred = pred_down
        current_gt = gt_down

    # Compute loss at the coarsest level.
    coarse_loss = torch.abs(current_pred - current_gt).mean()
    loss += (2 ** (levels - 1)) * coarse_loss

    return loss


def gradient_loss(
        pred: torch.Tensor,
        gt: torch.Tensor,
        use_grad_penalty: bool = False,
        grad_penalty_lambda: float = 0.0
) -> torch.Tensor:
    """
    Gradient loss that computes L1 difference on horizontal and vertical gradients.
    Optionally, a gradient penalty (scaled by grad_penalty_lambda) is added.
    """
    # Horizontal gradients.
    grad_pred_x = pred[..., :, 1:] - pred[..., :, :-1]
    grad_gt_x = gt[..., :, 1:] - gt[..., :, :-1]
    # Vertical gradients.
    grad_pred_y = pred[..., 1:, :] - pred[..., :-1, :]
    grad_gt_y = gt[..., 1:, :] - gt[..., :-1, :]

    loss = torch.abs(grad_pred_x - grad_gt_x).mean() + torch.abs(grad_pred_y - grad_gt_y).mean()

    if use_grad_penalty:
        penalty = (torch.abs(grad_pred_x).mean() + torch.abs(grad_pred_y).mean())
        loss = loss + grad_penalty_lambda * penalty

    return loss


class MattingLossV2(nn.Module):
    """
    Combined loss for image matting.

    Loss components:
      - Alpha L1 Loss
      - Composition Loss (if an image or (fg, bg) is provided)
      - Laplacian Pyramid Loss
      - Optional Gradient Loss
    """

    def __init__(
            self,
            weight_dict: Dict[str, float],
            device: torch.device,
            dtype: torch.dtype,
            use_grad_penalty: bool = False,
            grad_penalty_lambda: float = 0.0
    ):
        """
        Args:
            weight_dict (dict): Dictionary with keys 'l1', 'composition', 'laplacian', 'gradient'.
            device (torch.device): Device to store the loss tensors.
            dtype (torch.dtype): Data type for the loss tensors.
            use_grad_penalty (bool): Whether to include gradient penalty.
            grad_penalty_lambda (float): Scaling for gradient penalty.
        """
        super().__init__()

        # Normalize weights
        total_weight = sum(weight_dict.values())
        self.weight_dict = {k: v / total_weight for k, v in weight_dict.items()}
        for key in ["l1", "composition", "laplacian", "gradient"]:
            assert key in self.weight_dict, f"{key} loss weight must be provided."

        self.use_grad_penalty = use_grad_penalty
        self.grad_penalty_lambda = grad_penalty_lambda

        # Initialize buffers to store the losses
        self.register_buffer(name="l1_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="composition_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="laplacian_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="gradient_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name=CORE_LOSS_KEY, tensor=torch.tensor(data=0.0, dtype=dtype, device=device))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, image: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss.

        Args:
            pred (Tensor): Predicted alpha matte, shape (N,1,H,W).
            gt (Tensor): Ground truth alpha matte, shape (N,1,H,W).
            image (Tensor, optional): Input image, shape (N,C,H,W). Used for composition loss.

        Returns:
            Tensor: Total loss (scalar).
        """
        # Reset buffers
        self.l1_loss.zero_()
        self.composition_loss.zero_()
        self.laplacian_loss.zero_()
        self.gradient_loss.zero_()
        self.core_loss.zero_()

        # Compute losses
        losses = self._forward(pred, gt, image)

        # Update buffers
        self.l1_loss = losses["l1"].to(dtype=self.l1_loss.dtype, device=self.l1_loss.device)
        self.composition_loss = losses["composition"].to(
            dtype=self.composition_loss.dtype,
            device=self.composition_loss.device
        )
        self.laplacian_loss = losses["laplacian"].to(dtype=self.laplacian_loss.dtype, device=self.laplacian_loss.device)
        self.gradient_loss = losses["gradient"].to(dtype=self.gradient_loss.dtype, device=self.gradient_loss.device)
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

    def _forward(self, pred: torch.Tensor, gt: torch.Tensor, image: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the loss.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.
            image (torch.Tensor, optional): Input image for composition loss.


        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        losses = {"l1": 0, "composition": 0, "laplacian": 0, "gradient": 0}

        # Update losses
        self._update_losses(losses, pred, gt, image)

        # Reduce losses
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)

        return losses

    def _update_losses(
            self,
            losses: Dict[str, torch.Tensor],
            pred: torch.Tensor,
            gt: torch.Tensor,
            image: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update the loss components.

        Args:
            losses (Dict[str, torch.Tensor]): Dictionary of loss components.
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.
            image (torch.Tensor, optional): Input image for composition loss.

        Returns:
            Dict[str, torch.Tensor]: Updated dictionary of loss components.
        """
        # L1 loss
        losses['l1'] = l1_loss(pred, gt)

        # Composition loss
        if image is not None:
            losses['composition'] = composition_loss(pred, gt, image)

        # Laplacian pyramid loss.
        losses['laplacian'] = laplacian_loss(pred, gt)

        # Gradient loss.
        losses['gradient'] = gradient_loss(
            pred,
            gt,
            use_grad_penalty=self.use_grad_penalty,
            grad_penalty_lambda=self.grad_penalty_lambda
        )

        return losses


if __name__ == "__main__":
    # Generate some random predictions and ground truth
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = torch.rand((4, 1, 256, 256), dtype=dtype, requires_grad=True).to(device)
    targets = torch.rand((4, 1, 256, 256), dtype=dtype).to(device)
    images = torch.rand((4, 3, 256, 256), dtype=dtype).to(device)
    config = {"l1": 1.0, "composition": 1.0, "laplacian": 1.0, "gradient": 0.5}

    with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=dtype):
        # Compute the loss
        loss_fn = MattingLossV2(weight_dict=config, device=device, dtype=dtype)
        losses = loss_fn(predictions, targets, images)

        # Print the losses
        print(f"L1 Loss: {losses['l1']:.4f}")
        print(f"Composition Loss: {losses['composition']:.4f}")
        print(f"Laplacian Loss: {losses['laplacian']:.4f}")
        print(f"Gradient Loss: {losses['gradient']:.4f}")
        print(f"Core Loss: {losses[CORE_LOSS_KEY]:.4f}\n")

        # Print state_dict
        print(f"Loss function state dict: {loss_fn.state_dict()}")

    # Save state_dict
    torch.save(loss_fn.state_dict(), "matting_loss.pth")

    # Step 3: Create a new loss function instance
    new_loss_fn = MattingLossV2(weight_dict=config, device=device, dtype=dtype)

    # Step 4: Load the saved state dict into the new instance
    state_dict = torch.load("matting_loss.pth", weights_only=True)
    new_loss_fn.load_state_dict(state_dict)

    print("\nLoaded State Dict:")
    for key, value in new_loss_fn.state_dict().items():
        print(f"{key}: {value:.4f}")

    # Step 5: Verify that buffers match
    assert torch.allclose(loss_fn.l1_loss, new_loss_fn.l1_loss), "L1 loss mismatch"
    assert torch.allclose(loss_fn.composition_loss, new_loss_fn.composition_loss), "Composition loss mismatch"
    assert torch.allclose(loss_fn.laplacian_loss, new_loss_fn.laplacian_loss), "Laplacian loss mismatch"
    assert torch.allclose(loss_fn.gradient_loss, new_loss_fn.gradient_loss), "Gradient loss mismatch"
    assert torch.allclose(loss_fn.core_loss, new_loss_fn.core_loss), "Core loss mismatch"

    print("\nState dict successfully loaded and verified!")
