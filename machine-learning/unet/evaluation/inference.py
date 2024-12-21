import torch

from PIL import Image
import albumentations as A
import numpy as np
from typing import Dict
import time
import logging

from evaluation import metrics
from training.utils.train_utils import AverageMeter, MemMeter, DurationMeter, ProgressMeter


def _calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics for a batch of predictions and targets.

    Args:
        predictions (torch.Tensor): Predictions with shape (batch_size, ...).
        targets (torch.Tensor): Ground truth targets with shape (batch_size, ...).

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    # Initialize sums
    batch_size = predictions.size(0)
    mse_sum = 0.0
    sad_sum = 0.0
    grad_sum = 0.0

    # Compute metrics for each sample in the batch
    for i in range(batch_size):
        mse_sum += metrics.calculate_mse(predictions[i], targets[i])
        sad_sum += metrics.calculate_sad(predictions[i], targets[i])
        grad_sum += metrics.calculate_grad_error(predictions[i], targets[i])

    # Average metrics across the batch
    return {
        "mse": mse_sum / batch_size,
        "sad": sad_sum / batch_size,
        "grad": grad_sum / batch_size,
    }


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

    # Start timer
    start_time = time.time()

    # Init stat meters
    batch_time_meter = AverageMeter(name="Batch Time", device=str(device), fmt=":.2f")
    data_time_meter = AverageMeter(name="Data Time", device=str(device), fmt=":.2f")
    mem_meter = MemMeter(name="Mem (GB)", device=str(device), fmt=":.2f")
    time_elapsed_meter = DurationMeter(name="Time Elapsed", device=device, fmt=":.2f")

    # Progress bar
    iters_per_epoch = len(data_loader)
    progress = ProgressMeter(
        num_batches=iters_per_epoch,
        meters=[
            time_elapsed_meter,
            batch_time_meter,
            data_time_meter,
            mem_meter,
        ],
        real_meters={},
        prefix="Test | Epoch: [{}]".format(1),
    )

    # Model inference
    model.eval()
    metrics_sum = {}
    total_samples = 0
    end = time.time()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Measure data loading time
        data_time_meter.update(time.time() - end)

        # Move data to device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=torch.float16):
                    # Get predictions
                    outputs = model(X)
                    probabilities = torch.sigmoid(outputs)

                    # Calculate batch metrics
                    batch_results = _calculate_metrics(probabilities, y)
                    for k, v in batch_results.items():
                        metrics_sum[k] = metrics_sum.get(k, 0) + v
                    total_samples += len(y)

            # Measure elapsed time
            batch_time_meter.update(time.time() - end)
            end = time.time()
            time_elapsed_meter.update(time.time() - start_time)

            # Measure memory usage
            if torch.cuda.is_available():
                mem_meter.update(reset_peak_usage=True)

            # Update the progress bar every 10 batches
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise e

    # Compute average metrics
    metrics = {k: v / total_samples for k, v in metrics_sum.items()}
    logging.info(f"Evaluation completed")
    logging.info(f"Metrics: {metrics}")

    # Log state metrics
    state_metrics = {
        "data_time": data_time_meter.avg,
        "batch_time": batch_time_meter.avg,
        "mem": mem_meter.avg,
        "est_epoch_time": batch_time_meter.avg * iters_per_epoch,
    }

    logging.info(f"Batch Metrics: {state_metrics}")


def predict_image(
        image: Image.Image,
        model: torch.nn.Module,
        transform: A.Compose,
        device: torch.device,
) -> np.ndarray:
    """
    Predict the binary mask for a single image.

    Args:
        image (Image.Image): Input image
        model (torch.nn.Module): Model to use for prediction
        transform (albumentations.Compose): Transform to apply to the image
        device (torch.device): Device to use for inference

    Returns:
        np.ndarray: Predicted binary mask
    """
    model.eval()

    # Apply transformations
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            # Get predictions
            y_logits = model(image_tensor)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()  # Binary mask

    return preds.squeeze(0).cpu().numpy()  # Remove batch and convert to NumPy
