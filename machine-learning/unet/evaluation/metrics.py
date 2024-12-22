import torch
import torch.nn.functional as F


def compute_metrics_batch(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """
    Compute evaluation metrics for a batch of alpha matte predictions.

    Args:
        pred (torch.Tensor): Predicted alpha mattes (B, C, H, W).
        gt (torch.Tensor): Ground truth alpha mattes (B, C, H, W).

    Returns:
        dict[str, float]: Dictionary of averaged metrics for the batch.
    """
    # Initialize accumulators for metrics
    batch_metrics = {"mse": 0.0, "sad": 0.0, "grad": 0.0, "conn": 0.0}
    batch_size = pred.size(0)

    # Compute metrics for each sample in the batch
    for i in range(batch_size):
        sample_metrics = compute_metrics(pred[i:i+1], gt[i:i+1])  # Process one sample at a time
        for key in batch_metrics:
            batch_metrics[key] += sample_metrics[key]

    # Average metrics over the batch
    for key in batch_metrics:
        batch_metrics[key] /= batch_size

    return batch_metrics


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """
    Compute evaluation metrics for alpha matte prediction.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        dict[str, float]: Dictionary of evaluation metrics.
    """
    metrics = {
        "mse": calculate_mse(pred, gt),
        "sad": calculate_sad(pred, gt),
        "grad": calculate_grad_error(pred, gt),
        "conn": connectivity_error(pred, gt)
    }

    return metrics


def calculate_mse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error (MSE) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: MSE value.
    """
    mse = F.mse_loss(pred, gt, reduction="mean")
    return mse.item()


def calculate_sad(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Calculate Sum of Absolute Differences (SAD) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: SAD value.
    """
    sad = torch.sum(torch.abs(pred - gt)) / pred.numel()  # Normalize by total pixels
    return sad.item()


def calculate_grad_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Calculate Gradient Error (GRAD) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.

    Returns:
        float: GRAD value.
    """
    # Sobel filters for gradient computation
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    # Gradients
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_gt_x = F.conv2d(gt, sobel_x, padding=1)
    grad_gt_y = F.conv2d(gt, sobel_y, padding=1)

    # Magnitude differences
    grad_diff = torch.abs(grad_pred_x - grad_gt_x) + torch.abs(grad_pred_y - grad_gt_y)
    grad_error = torch.sum(grad_diff) / pred.numel()  # Normalize by total pixels

    return grad_error.item()


def connectivity_error(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.1) -> float:
    """
    Calculate Connectivity Error (CONN) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted alpha matte.
        gt (torch.Tensor): Ground truth alpha matte.
        threshold (float): Threshold for boundary detection.

    Returns:
        float: CONN value.
    """
    # Threshold alpha mattes
    pred_boundary = (pred > threshold).float()
    gt_boundary = (gt > threshold).float()

    # Connected regions difference
    conn_diff = torch.abs(pred_boundary - gt_boundary)
    conn = torch.sum(conn_diff) / gt.numel()  # Normalize by total pixels

    return conn.item()
