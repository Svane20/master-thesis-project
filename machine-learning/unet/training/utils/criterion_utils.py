from typing import List

import torch
import torch.nn.functional as F


def compute_boundary_map(gt: torch.Tensor, threshold: float = 0.1, epsilon: float = 1e-6) -> torch.Tensor:
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


def laplacian_pyramid(image: torch.Tensor, kernel: torch.Tensor, max_levels: int) -> List[torch.Tensor]:
    """
    Create a Laplacian pyramid from an input image.

    Args:
        image (torch.Tensor): Input image to create the pyramid from.
        kernel (torch.Tensor): Gaussian kernel to use.
        max_levels (int): Number of levels in the pyramid.

    Returns:
        List[torch.Tensor]: List of images in the Laplacian pyramid.
    """
    pyramid = []

    # Create Gaussian pyramid
    current = image
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down

    return pyramid


def gauss_kernel(device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.

    Args:
        device (torch.device): Device to create the kernel on. Default is 'cpu'.
        dtype (torch.dtype): Data type of the kernel. Default is 'torch.float32'.

    Returns:
        torch.Tensor: 2D Gaussian kernel.
    """
    kernel = torch.tensor(
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]],
        dtype=dtype, device=device
    ) / 256.0

    return kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 5, 5]


def gauss_conv2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Apply a 2D Gaussian convolution to an input tensor.

    Args:
        image (torch.Tensor): Input tensor to convolve.
        kernel (torch.Tensor): Gaussian kernel to use.

    Returns:
        torch.Tensor: Convolved tensor.
    """
    B, C, H, W = image.shape

    # Reshape input tensor for convolution
    image = image.reshape(B * C, 1, H, W)

    # Pad input tensor
    image = F.pad(input=image, pad=(2, 2, 2, 2), mode='reflect')

    # Apply convolution
    output = F.conv2d(image, kernel)

    # Reshape output tensor
    return output.reshape(B, C, H, W)


def downsample(image: torch.tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Downsample an image by applying a 2D Gaussian convolution followed by strided pooling.

    Args:
        image (torch.Tensor): Input image to downsample.
        kernel (torch.Tensor): Gaussian kernel to use.

    Returns:
        torch.Tensor: Downsampled image.
    """
    # Apply Gaussian convolution
    image = gauss_conv2d(image, kernel)

    # Apply strided pooling
    return image[:, :, ::2, ::2]


def upsample(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Upsample an image by applying a 2D Gaussian convolution followed by transposed convolution.

    Args:
        image (torch.Tensor): Input image to upsample.
        kernel (torch.Tensor): Gaussian kernel to use.

    Returns:
        torch.Tensor: Upsampled image.
    """
    B, C, H, W = image.shape

    # Create output tensor
    output = torch.zeros((B, C, H * 2, W * 2), device=image.device, dtype=image.dtype)

    # Upsample using transposed convolution
    output[:, :, ::2, ::2] = image * 4

    # Apply Gaussian convolution
    return gauss_conv2d(output, kernel)


def crop_to_even_size(image: torch.Tensor) -> torch.Tensor:
    """
    Crop an image to an even size.

    Args:
        image (torch.Tensor): Input image to crop.

    Returns:
        torch.Tensor: Cropped image.
    """
    _, _, H, W = image.shape

    # Ensure the image dimensions are even
    H = H - H % 2
    W = W - W % 2

    # Crop the image to an even size
    return image[:, :, :H, :W]
