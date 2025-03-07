import torch

# Sobel Kernels
SOBEL_KERNEL_X = torch.tensor(
    [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
).view(1, 1, 3, 3)

SOBEL_KERNEL_Y = torch.tensor(
    [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
).view(1, 1, 3, 3)


def compute_edge_map(tensor: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the edge map of a binary mask using the Sobel filter.

    Args:
        tensor (torch.Tensor): Binary mask tensor.
        epsilon (float): Small value to avoid division by zero. Default is 1e-6.

    Returns:
        torch.Tensor: Edge map tensor.
    """
    device, dtype = tensor.device, tensor.dtype
    sobel_x = SOBEL_KERNEL_X.to(device=device, dtype=dtype)
    sobel_y = SOBEL_KERNEL_Y.to(device=device, dtype=dtype)

    # Compute the gradient in the x and y directions
    grad_x = torch.nn.functional.conv2d(tensor, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(tensor, sobel_y, padding=1)

    # Compute the magnitude of the gradient
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)

    return grad_magnitude
