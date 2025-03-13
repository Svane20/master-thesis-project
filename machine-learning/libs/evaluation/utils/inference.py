import torch

import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging

from libs.datasets.transforms import Transform
from libs.metrics.utils import compute_evaluation_metrics, get_grad_filter
from libs.training.utils.train_utils import AverageMeter, ProgressMeter


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> None:
    """
    Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
    """
    if len(data_loader) == 0:
        logging.warning("DataLoader is empty. No evaluation performed.")
        return

    # Set model to evaluation mode
    model.eval()

    # Gradient filter
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    grad_filter = get_grad_filter(device=device, dtype=dtype)

    # Metric meters
    mse_meter = AverageMeter(name="MSE", device=str(device), fmt=":.2e")
    mae_meter = AverageMeter(name="MAE", device=str(device), fmt=":.2e")
    sad_meter = AverageMeter(name="SAD", device=str(device), fmt=":.2e")
    grad_meter = AverageMeter(name="Grad", device=str(device), fmt=":.2e")
    conn_meter = AverageMeter(name="Conn", device=str(device), fmt=":.2e")
    metrics = {}

    # Progress bar
    progress = ProgressMeter(
        num_batches=len(data_loader),
        meters=[
            mse_meter,
            mae_meter,
            sad_meter,
            grad_meter,
            conn_meter,
        ],
        real_meters={},
        prefix="Test | Epoch: [{}]".format(1),
    )

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data to device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        try:
            with torch.no_grad():
                with torch.amp.autocast(
                        device_type=device.type,
                        enabled=torch.cuda.is_available(),
                        dtype=dtype
                ):
                    # Get predictions
                    outputs = model(X)

                    # Calculate metrics
                    scores = compute_evaluation_metrics(outputs, y, grad_filter)

                    # Accumulate metrics
                    batch_size = y.size(0)
                    mse_meter.update(scores["mse"], n=batch_size)
                    mae_meter.update(scores["mae"], n=batch_size)
                    sad_meter.update(scores["sad"], n=batch_size)
                    grad_meter.update(scores["grad"], n=batch_size)
                    conn_meter.update(scores["conn"], n=batch_size)

            # Update the progress bar every 10 batches
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e

    # Logs metrics
    metrics["mse"] = mse_meter.avg
    metrics["mae"] = mae_meter.avg
    metrics["grad"] = grad_meter.avg
    metrics["sad"] = sad_meter.avg
    metrics["conn"] = conn_meter.avg

    logging.info("Evaluation finished.")
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")


def predict_image(
        image: Image,
        mask: Image,
        model: torch.nn.Module,
        transform: Transform,
        device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, float]]:
    """
    Predict the alpha matte for an image.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Ground truth mask.
        model (torch.nn.Module): Model to use for prediction.
        transform (Transform): Transform to apply to the image.
        device (torch.device): Device to use for evaluation.

    Returns:
        Tuple[torch.Tensor, np.ndarray, Dict[str, float]]: Raw output, predicted mask and evaluation metrics.
    """
    # Set model to evaluation mode
    model.eval()

    # Gradient filter
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    grad_filter = get_grad_filter(device=device, dtype=dtype)

    # Metric meters
    mse_meter = AverageMeter(name="MSE", device=str(device), fmt=":.2e")
    mae_meter = AverageMeter(name="MAE", device=str(device), fmt=":.2e")
    sad_meter = AverageMeter(name="SAD", device=str(device), fmt=":.2e")
    grad_meter = AverageMeter(name="Grad", device=str(device), fmt=":.2e")
    conn_meter = AverageMeter(name="Conn", device=str(device), fmt=":.2e")
    metrics = {}

    # Apply transformations
    image_tensor, mask_tensor = transform(image, mask)

    # Move tensors to device and add batch dimension
    image_tensor, mask_tensor = image_tensor.unsqueeze(0).to(device), mask_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=torch.float16):
            outputs = model(image_tensor)

            # Calculate metrics
            scores = compute_evaluation_metrics(outputs, mask_tensor, grad_filter)

            # Accumulate metrics
            batch_size = mask_tensor.size(0)
            mse_meter.update(scores["mse"], n=batch_size)
            mae_meter.update(scores["mae"], n=batch_size)
            sad_meter.update(scores["sad"], n=batch_size)
            grad_meter.update(scores["grad"], n=batch_size)
            conn_meter.update(scores["conn"], n=batch_size)

    # Logs metrics
    metrics["mse"] = mse_meter.avg
    metrics["mae"] = mae_meter.avg
    metrics["grad"] = grad_meter.avg
    metrics["sad"] = sad_meter.avg
    metrics["conn"] = conn_meter.avg

    logging.info(f"Metrics")
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")

    # Convert alpha matte to numpy array
    pred_alpha = outputs.squeeze(0).cpu().numpy()

    return outputs, pred_alpha, metrics
