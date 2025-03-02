import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List

CORE_LOSS_KEY = 'core_loss'


class MRSDLoss(nn.Module):
    """
    Mean-root square difference loss.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """
        Args:
            eps (float, optional): Epsilon. Defaults to 1e-6.
        """
        super().__init__()

        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the MRSD loss.

        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: MRSD loss.
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        diff_sq = (pred - target) ** 2
        loss = torch.sqrt(diff_sq + self.eps)

        return loss.mean()


class GradientLoss(nn.Module):
    """
    Gradient loss based on the Sobel filter.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, eps: float = 1e-6) -> None:
        """
        Args:
            device (torch.device): Device for loss computations.
            dtype (torch.dtype): Datatype for computation.
            eps (float, optional): Epsilon. Defaults to 1e-6.
        """
        super().__init__()

        self.eps = eps

        # Define Sobel kernels for x and y directions
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=dtype, device=device)
        sobel_y = sobel_x.t()
        sobel_x = sobel_x / sobel_x.abs().sum()
        sobel_y = sobel_y / sobel_y.abs().sum()

        # Reshape to (out_channels, in_channels, kH, kW)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient loss.

        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Gradient loss.
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        grad_pred = self._compute_sobel(pred)
        grad_target = self._compute_sobel(target)

        return F.l1_loss(grad_pred, grad_target, reduction='mean')

    def _compute_sobel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient using Sobel filters.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Gradient tensor.
        """
        # Replicate pad 1 pixel on each side
        input_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(input_padded, self.sobel_x)
        grad_y = F.conv2d(input_padded, self.sobel_y)

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)


class LaplacianLoss(nn.Module):
    """
    Laplacian loss based on the Laplacian pyramid.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, size: int = 5, sigma: float = 1.0,
                 max_levels: int = 5) -> None:
        """
        Args:
            device (torch.device): Device for loss computations.
            dtype (torch.dtype): Data type for loss computations.
            size (int, optional): Size of the Gaussian kernel. Defaults to 5.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            max_levels (int, optional): Maximum number of levels in the Laplacian pyramid. Defaults to 5.
        """
        super().__init__()

        self.max_levels = max_levels

        kernel = self._build_gauss_kernel(size, sigma, n_channels=1)
        if device is not None:
            kernel = kernel.to(device=device, dtype=dtype)

        self.register_buffer("gauss_kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Laplacian loss.

        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Laplacian loss.
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        pyr_pred = self._laplacian_pyramid(pred, self.gauss_kernel, self.max_levels)
        pyr_target = self._laplacian_pyramid(target, self.gauss_kernel, self.max_levels)

        loss = 0.0
        for p, t in zip(pyr_pred, pyr_target):
            loss += F.l1_loss(p, t, reduction='mean')

        return loss

    def _build_gauss_kernel(self, size: int = 5, sigma: float = 1.0, n_channels: int = 1) -> torch.Tensor:
        """
        Build a Gaussian kernel.

        Args:
            size (int, optional): Size of the kernel. Defaults to 5.
            sigma (float, optional): Standard deviation of the kernel. Defaults to 1.0.
            n_channels (int, optional): Number of channels. Defaults to 1.

        Returns:
            torch.Tensor: Gaussian kernel.
        """
        # Ensure kernel size is odd
        if size % 2 != 1:
            raise ValueError("Kernel size must be odd")

        # Create a coordinate grid using torch.arange and torch.meshgrid
        coords = torch.arange(size, dtype=torch.float32)

        # Create a grid of shape (size, size)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        center = size // 2

        # Compute the Gaussian values as in the PaddlePaddle version:
        # exp((x - center)**2 / (-2*sigma**2))**2 = exp((x-center)**2 / (-sigma**2))
        g_x = torch.exp(((x - center) ** 2) / (-sigma ** 2))
        g_y = torch.exp(((y - center) ** 2) / (-sigma ** 2))

        # Sum the contributions from both dimensions
        kernel = g_x + g_y
        kernel = kernel / torch.sum(kernel)

        # Reshape to (n_channels, 1, size, size)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(n_channels, 1, 1, 1)

        return kernel

    def _conv_gauss(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Convolve the input tensor with a Gaussian kernel.

        Args:
            x (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Gaussian kernel.

        Returns:
            torch.Tensor: Convolved tensor.
        """
        n_channels, _, kh, kw = kernel.shape

        pad_h = kh // 2
        pad_w = kw // 2
        input_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')

        return F.conv2d(input_pad, kernel, groups=n_channels)

    def _laplacian_pyramid(self, x: torch.Tensor, kernel: torch.Tensor, max_levels: int = 5) -> List[torch.Tensor]:
        """
        Compute the Laplacian pyramid.

        Args:
            x (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Gaussian kernel.
            max_levels (int, optional): Maximum number of levels in the pyramid. Defaults to 5.

        Returns:
            List[torch.Tensor]: List of tensors representing the Laplacian pyramid.
        """
        current = x
        pyramid = []

        for level in range(max_levels):
            filtered = self._conv_gauss(current, kernel)
            diff = current - filtered
            pyramid.append(diff)
            current = F.avg_pool2d(filtered, kernel_size=2)

        pyramid.append(current)

        return pyramid


class MattingLossV2(nn.Module):
    """
    Combined loss function for alpha matting tasks using MRSD, gradient, and Laplacian losses.
    """

    def __init__(self, weight_dict: Dict[str, float], device: torch.device, dtype: torch.dtype) -> None:
        """
        Args:
            weight_dict (Dict[str, float]): Dictionary with keys "mrsd", "gradient", and "laplacian" indicating their weights.
            device (torch.device): Device for loss computations.
            dtype (torch.dtype): Data type for loss computations.
        """
        super().__init__()

        # Check that all keys exist
        for key in ["reconstruction", "gradient", "laplacian"]:
            if key not in weight_dict:
                raise ValueError(f"Key '{key}' not found in weight_dict")

        # Normalize weights
        total_weight = sum(weight_dict.values())
        self.weight_dict = {k: v / total_weight for k, v in weight_dict.items()}

        # Loss functions
        self.reconstruction_loss_fn = MRSDLoss()
        self.gradient_loss_fn = GradientLoss(dtype=dtype, device=device)
        self.laplacian_loss_fn = LaplacianLoss(dtype=dtype, device=device)

        # Register buffers for monitoring individual loss components
        self.register_buffer("reconstruction_loss", torch.tensor(0.0, dtype=dtype, device=device))
        self.register_buffer("gradient_loss", torch.tensor(0.0, dtype=dtype, device=device))
        self.register_buffer("laplacian_loss", torch.tensor(0.0, dtype=dtype, device=device))
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
        self.gradient_loss.zero_()
        self.laplacian_loss.zero_()
        self.core_loss.zero_()

        # Compute losses
        losses = self._forward(pred, gt)

        # Update buffers
        self.reconstruction_loss = losses["reconstruction"].to(
            dtype=self.reconstruction_loss.dtype,
            device=self.reconstruction_loss.device
        )
        self.gradient_loss = losses["gradient"].to(dtype=self.gradient_loss.dtype, device=self.gradient_loss.device)
        self.laplacian_loss = losses["laplacian"].to(dtype=self.laplacian_loss.dtype, device=self.laplacian_loss.device)
        self.core_loss = losses[CORE_LOSS_KEY].to(dtype=self.core_loss.dtype, device=self.core_loss.device)

        return losses

    def _forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the loss.

        Args:
            pred (torch.Tensor): Predicted alpha matte.
            gt (torch.Tensor): Ground truth alpha matte.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        losses = {"reconstruction": 0, "gradient": 0, "laplacian": 0}

        # Update losses
        self._update_losses(losses, pred, gt)

        # Reduce losses
        losses[CORE_LOSS_KEY] = self._reduce_loss(losses)

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
        # Compute L1 loss
        losses["reconstruction"] = self.reconstruction_loss_fn(pred, gt)

        # Compute gradient loss
        losses["gradient"] = self.gradient_loss_fn(pred, gt)

        # Laplacian loss
        losses["laplacian"] = self.laplacian_loss_fn(pred, gt)

        return losses

    def _reduce_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
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


def check_loss_devices(loss_fn: MattingLossV2):
    # MRSDLoss doesn't register any buffers, so we assume it uses the input device.
    print("MRSDLoss: (uses input device)")

    # For GradientLoss, check one of the Sobel buffers.
    print("GradientLoss sobel_x device:", loss_fn.gradient_loss_fn.sobel_x.device)
    print("GradientLoss sobel_x dtype: ", loss_fn.gradient_loss_fn.sobel_x.dtype)

    # For LaplacianLoss, check the Gaussian kernel buffer.
    print("LaplacianLoss gauss_kernel device:", loss_fn.laplacian_loss_fn.gauss_kernel.device)
    print("LaplacianLoss gauss_kernel dtype: ", loss_fn.laplacian_loss_fn.gauss_kernel.dtype)

    # Also, check the monitoring buffers in MattingLoss.
    print("MattingLoss reconstruction_loss device:", loss_fn.reconstruction_loss.device)
    print("MattingLoss reconstruction_loss dtype: ", loss_fn.reconstruction_loss.dtype)
    print("MattingLoss gradient_loss device:", loss_fn.gradient_loss.device)
    print("MattingLoss gradient_loss dtype: ", loss_fn.gradient_loss.dtype)
    print("MattingLoss laplacian_loss device:", loss_fn.laplacian_loss.device)
    print("MattingLoss laplacian_loss dtype: ", loss_fn.laplacian_loss.dtype)
    print("MattingLoss total_loss device:", getattr(loss_fn, CORE_LOSS_KEY).device)
    print("MattingLoss total_loss dtype: ", getattr(loss_fn, CORE_LOSS_KEY).dtype)


if __name__ == '__main__':
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate predictions and ground truth alpha mattes of shape [B, 1, H, W].
    pred_alpha = torch.rand((4, 1, 256, 256), dtype=dtype, device=device, requires_grad=True)
    gt_alpha = torch.rand((4, 1, 256, 256), dtype=dtype, device=device)

    # Define loss weights for each component.
    config = {"reconstruction": 1.0, "gradient": 0.5, "laplacian": 0.5}

    loss_fn = MattingLossV2(weight_dict=config, device=device, dtype=dtype)
    losses = loss_fn(pred_alpha, gt_alpha)
    # check_loss_devices(loss_fn)

    print(f"Reconstruction Loss: {losses['reconstruction']:.4f}")
    print(f"Gradient Loss: {losses['gradient']:.4f}")
    print(f"Laplacian Loss: {losses['laplacian']:.4f}")
    print(f"Total Loss: {losses[CORE_LOSS_KEY]:.4f}")
