import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from libs.training.utils.criterion_utils import CriterionBase, BaseCriterionConfig, register_loss

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


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    target_pyramid = laplacian_pyramid(target, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], target_pyramid[level])
    return loss / max_levels


def laplacian_pyramid(img, kernel, max_levels):
    pyramid = []
    current = img
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid


def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], device=device, dtype=dtype) / 256
    return kernel.view(1, 1, 5, 5)


def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.view(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode="reflect")
    img = F.conv2d(img, kernel)
    return img.view(B, C, H, W)


def downsample(img, kernel):
    return gauss_convolution(img, kernel)[:, :, ::2, ::2]


def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    return gauss_convolution(out, kernel)


def crop_to_even_size(img):
    H, W = img.shape[2:]
    return img[:, :, :H - H % 2, :W - W % 2]


def gradient_loss(
        pred: torch.Tensor,
        gt: torch.Tensor,
        use_grad_penalty: bool = False,
        grad_penalty_lambda: float = 0.0
) -> torch.Tensor:
    """
    Gradient loss that computes the L1 difference on Sobel-filtered gradients.
    Optionally, a gradient penalty (scaled by grad_penalty_lambda) is added.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.
        use_grad_penalty (bool): If True, adds a gradient penalty term.
        grad_penalty_lambda (float): Scaling factor for the gradient penalty.

    Returns:
        torch.Tensor: Computed gradient loss.
    """
    # Sobel filters for computing gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    # Compute gradients for prediction and ground truth using Sobel filters.
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_gt_x = F.conv2d(gt, sobel_x, padding=1)
    grad_gt_y = F.conv2d(gt, sobel_y, padding=1)

    # Compute the L1 difference between the corresponding gradients.
    loss_x = torch.abs(grad_pred_x - grad_gt_x)
    loss_y = torch.abs(grad_pred_y - grad_gt_y)
    loss = torch.mean(loss_x + loss_y)

    # Optionally add a gradient penalty term based on the predicted gradients.
    if use_grad_penalty:
        penalty = torch.mean(torch.abs(grad_pred_x)) + torch.mean(torch.abs(grad_pred_y))
        loss = loss + grad_penalty_lambda * penalty

    return loss


def boundary_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Boundary-aware loss that weights the L1 loss based on the boundary map.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        torch.Tensor: Computed boundary-aware loss.
    """
    boundary_map = _compute_boundary_map(gt)

    l1 = F.l1_loss(pred, gt, reduction='none')

    # Weight the L1 loss based on the boundary map.
    weight = boundary_map + (((gt > 0.0) & (gt < 1.0)).float())
    loss = l1 * weight

    return loss.mean()


def _compute_boundary_map(gt: torch.Tensor, threshold: float = 0.1, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute a boundary map from the ground truth alpha matte.

    Args:
        gt (torch.Tensor): Ground truth alpha matte.
        threshold (float): Threshold for boundary detection. Default is 0.1.
        epsilon (float): Small epsilon for numerical stability. Default is 1e-6.

    Returns:
        torch.Tensor: Boundary map.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gt.dtype, device=gt.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gt.dtype, device=gt.device).view(1, 1, 3, 3)

    grad_gt_x = F.conv2d(gt, sobel_x, padding=1)
    grad_gt_y = F.conv2d(gt, sobel_y, padding=1)

    grad_magnitude = torch.sqrt(grad_gt_x ** 2 + grad_gt_y ** 2 + epsilon)
    boundary_map = (grad_magnitude > threshold).float()

    return boundary_map


class MattingCriterion(CriterionBase):
    SUPPORTED_LOSSES = {
        "gradient_loss",
        "unknown_l1_loss",
        "known_l1_loss",
        "laplacian_pha_loss"
    }

    def __init__(self, config: BaseCriterionConfig, device: torch.device) -> None:
        super().__init__(config=config, device=device)

    def forward(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        return super().forward(preds, targets, **kwargs)

    # Loss Functions
    @register_loss("gradient_loss", ("sample_map",))
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, sample_map: torch.Tensor = None) -> torch.Tensor:
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=pred.device, dtype=pred.dtype)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=pred.device, dtype=pred.dtype)

        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)

        if sample_map is not None:
            scale = sample_map.shape[0] * 262144 / torch.sum(sample_map)
            return (
                    F.l1_loss(grad_pred_x * sample_map, grad_target_x * sample_map) +
                    F.l1_loss(grad_pred_y * sample_map, grad_target_y * sample_map) +
                    0.01 * torch.mean(torch.abs(grad_pred_x * sample_map)) +
                    0.01 * torch.mean(torch.abs(grad_pred_y * sample_map))
            ) * scale

        return (
                F.l1_loss(grad_pred_x, grad_target_x) +
                F.l1_loss(grad_pred_y, grad_target_y) +
                0.01 * torch.mean(torch.abs(grad_pred_x)) +
                0.01 * torch.mean(torch.abs(grad_pred_y))
        )

    @register_loss("unknown_l1_loss", ("sample_map",))
    def unknown_l1_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_map: torch.Tensor = None
    ) -> torch.Tensor:
        if sample_map is not None:
            scale = sample_map.shape[0] * 262144 / torch.sum(sample_map)
            return F.l1_loss(pred * sample_map, target * sample_map) * scale
        return F.l1_loss(pred, target)

    @register_loss("known_l1_loss", ("sample_map",))
    def known_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, sample_map: torch.Tensor = None) -> torch.Tensor:
        if sample_map is not None:
            known_map = (sample_map == 0).float()
            scale = known_map.shape[0] * 262144 / torch.sum(known_map) if torch.sum(known_map) > 0 else 0
            return F.l1_loss(pred * known_map, target * known_map) * scale
        return F.l1_loss(pred, target)

    @register_loss("laplacian_pha_loss", ())
    def laplacian_pha_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return laplacian_loss(pred, target)


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
        for key in ["boundary", "composition", "gradient", "l1", "laplacian"]:
            assert key in self.weight_dict, f"{key} loss weight must be provided."

        self.use_grad_penalty = use_grad_penalty
        self.grad_penalty_lambda = grad_penalty_lambda

        # Initialize buffers to store the losses
        self.register_buffer(name="boundary_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="composition_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="gradient_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="l1_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
        self.register_buffer(name="laplacian_loss", tensor=torch.tensor(data=0.0, dtype=dtype, device=device))
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
        self.boundary_loss.zero_()
        self.composition_loss.zero_()
        self.gradient_loss.zero_()
        self.l1_loss.zero_()
        self.laplacian_loss.zero_()
        self.core_loss.zero_()

        # Compute losses
        losses = self._forward(pred, gt, image)

        # Update buffers
        self.boundary_loss = losses["boundary"].to(dtype=self.boundary_loss.dtype, device=self.boundary_loss.device)
        self.composition_loss = losses["composition"].to(
            dtype=self.composition_loss.dtype,
            device=self.composition_loss.device
        )
        self.gradient_loss = losses["gradient"].to(dtype=self.gradient_loss.dtype, device=self.gradient_loss.device)
        self.l1_loss = losses["l1"].to(dtype=self.l1_loss.dtype, device=self.l1_loss.device)
        self.laplacian_loss = losses["laplacian"].to(dtype=self.laplacian_loss.dtype, device=self.laplacian_loss.device)
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
        losses = {"boundary": 0, "composition": 0, "gradient": 0, "l1": 0, "laplacian": 0}

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
        # Boundary loss
        losses['boundary'] = boundary_loss(pred, gt)

        # Composition loss
        if image is not None:
            losses['composition'] = composition_loss(pred, gt, image)

        # Gradient loss.
        losses['gradient'] = gradient_loss(
            pred,
            gt,
            use_grad_penalty=self.use_grad_penalty,
            grad_penalty_lambda=self.grad_penalty_lambda
        )

        # L1 loss
        losses['l1'] = l1_loss(pred, gt)

        # Laplacian pyramid loss.
        losses['laplacian'] = laplacian_loss(pred, gt)

        return losses


if __name__ == "__main__":
    # Generate some random predictions and ground truth
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = torch.rand((4, 1, 256, 256), dtype=dtype, requires_grad=True).to(device)
    targets = torch.rand((4, 1, 256, 256), dtype=dtype).to(device)
    images = torch.rand((4, 3, 256, 256), dtype=dtype).to(device)
    trimap = torch.randint(0, 256, (4, 256, 256), dtype=dtype).to(device)

    # Initialize the loss function
    criterion = MattingCriterion(
        BaseCriterionConfig(
            losses=["unknown_l1_loss", "known_l1_loss", "gradient_loss", "laplacian_pha_loss"],
            weight_dict={
                "unknown_l1_loss": 1.0,
                "known_l1_loss": 0.3,
                "gradient_loss": 0.25,
                "laplacian_pha_loss": 0.5
            },
            normalize_weights=True
        ),
        device=device
    )
    print(f"Criterion: {criterion}")

    # Compute the loss
    losses = criterion(predictions, targets, trimap=trimap)
    print(f"Losses: {losses}")

    # Compute the logs
    logs = criterion.log_dict()
    print(f"Logs: {logs}")

    # Compute the total loss
    total_loss = losses["core_loss"]
    print("Total loss:", total_loss)

    # Print criterion state dict
    print(f"Criterion state dict: {criterion.state_dict()}")

    # Save the loss
    torch.save({
        "state_dict": criterion.state_dict(),
        "config": criterion.get_config(),
    }, "matting_criterion.pth")

    # Load the loss function
    checkpoint = torch.load("matting_criterion.pth")
    loaded_criterion = MattingCriterion.from_config(checkpoint["config"], device)
    loaded_criterion.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded state dict: {loaded_criterion.state_dict()}")
    print(f"Loaded config: {loaded_criterion.get_config()}")


    def assert_state_dicts_close(dict1, dict2, atol=1e-3, rtol=1e-3):
        for k in dict1:
            assert k in dict2, f"Missing key in second state_dict: {k}"
            v1 = dict1[k]
            v2 = dict2[k]

            # Make sure they are both tensors and cast to float32 for comparison
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if not torch.allclose(v1.float(), v2.float(), atol=atol, rtol=rtol):
                    raise AssertionError(f"Mismatch at {k}:\n  {v1} !=\n  {v2}")
            else:
                if v1 != v2:
                    raise AssertionError(f"Mismatch at {k}: {v1} != {v2}")


    # Assert that the loaded loss function is the same
    assert_state_dicts_close(criterion.state_dict(), loaded_criterion.state_dict())
