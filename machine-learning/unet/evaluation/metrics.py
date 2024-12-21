import torch
import torch.nn.functional as F


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error (MSE) between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted alpha matte.
        targets (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: MSE value.
    """
    mse = torch.mean((predictions - targets) ** 2)
    return mse.item()


def calculate_sad(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Sum of Absolute Differences (SAD) between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted alpha matte.
        targets (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: SAD value.
    """
    sad = torch.sum(torch.abs(predictions - targets))
    return sad.item()


def calculate_grad_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Gradient Error (GRAD) between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted alpha matte.
        targets (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: GRAD value.
    """
    # Ensure predictions and targets have proper shape
    predictions = predictions.squeeze(1)  # Remove singleton dimension if present
    targets = targets.squeeze(1)

    # Add channel dimension for conv2d
    predictions = predictions.unsqueeze(1)
    targets = targets.unsqueeze(1)

    # Define Sobel filters for gradient computation
    sobel_x = torch.tensor(
        data=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=predictions.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=predictions.device
    ).view(1, 1, 3, 3)

    # Apply Sobel filters to compute gradients
    grad_pred_x = F.conv2d(predictions, sobel_x, padding=1)
    grad_pred_y = F.conv2d(predictions, sobel_y, padding=1)
    grad_target_x = F.conv2d(targets, sobel_x, padding=1)
    grad_target_y = F.conv2d(targets, sobel_y, padding=1)

    # Calculate gradient magnitudes
    grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2)
    grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2)

    # Compute gradient error
    grad_error = torch.sum(torch.abs(grad_pred - grad_target))
    return grad_error.item()
