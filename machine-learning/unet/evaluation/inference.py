import torch
import torchvision

import albumentations as A
import numpy as np
from typing import Dict, Tuple
import time
import logging

from metrics.utils import compute_metrics, compute_evaluation_metrics
from training.utils.train_utils import AverageMeter, MemMeter, DurationMeter, ProgressMeter


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
    end = time.time()

    # Metrics
    metrics_sum = {"sad": 0, "mse": 0, "mae": 0, "grad": 0, "conn": 0}
    total_samples = 0

    for batch_idx, (X, y) in enumerate(data_loader):
        # Measure data loading time
        data_time_meter.update(time.time() - end)

        # Move data to device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        try:
            with torch.no_grad():
                with torch.amp.autocast(
                        device_type=device.type,
                        enabled=torch.cuda.is_available(),
                        dtype=torch.float16
                ):
                    # Get predictions
                    outputs = model(X)

                    # Clamp outputs to ensure they're in [0, 1]
                    outputs = torch.clamp(outputs, 0, 1)

                    # Calculate batch metrics
                    batch_metrics = compute_evaluation_metrics(outputs, y)

                    for k, v in batch_metrics.items():
                        metrics_sum[k] += v * X.size(0)

                    total_samples += X.size(0)

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
    avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
    logging.info(f"Evaluation completed")
    logging.info(f"Metrics: {avg_metrics}")

    # Log state metrics
    state_metrics = {
        "data_time": data_time_meter.avg,
        "batch_time": batch_time_meter.avg,
        "mem": mem_meter.avg,
        "est_epoch_time": batch_time_meter.avg * iters_per_epoch,
    }

    logging.info(f"Batch Metrics: {state_metrics}")


def predict_image(
        image: np.ndarray,
        mask: np.ndarray,
        model: torch.nn.Module,
        transform: A.Compose,
        device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Predict the alpha matte for an image.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Ground truth mask.
        model (torch.nn.Module): Model to use for prediction.
        transform (albumentations.Compose): Transform to apply to the image.
        device (torch.device): Device to use for inference.

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: Predicted mask and evaluation metrics.
    """
    model.eval()

    # Apply transformations
    transformed = transform(image=image, mask=mask)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension
    mask_tensor = transformed["mask"].unsqueeze(0).to(device)  # Add batch dimension

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=torch.float16):
            outputs = model(image_tensor)
            outputs = torch.clamp(outputs, 0, 1)  # Clamp to [0, 1]

            # Calculate metrics
            metrics = compute_metrics(outputs, mask_tensor, is_eval=True)

    logging.info("Metrics:")
    for k, v in metrics.items():
        logging.info(f"\t{k}: {v}")

    # Convert alpha matte to numpy array
    pred_alpha = outputs.squeeze(0).cpu().numpy()

    # Validate alpha matte values
    validation_results = _validate_alpha_matte(pred_alpha)
    logging.info(f"Alpha Matte Validation Results: {validation_results}")

    # Save the predicted mask
    torchvision.utils.save_image(outputs, "predictions/output.png")

    return pred_alpha, metrics


def _validate_alpha_matte(alpha_matte: np.ndarray, tolerance: float = 0.05) -> Dict[str, bool]:
    """
    Validate the alpha matte for expected values (0, 0.5, 1).

    Args:
        alpha_matte (np.ndarray): Predicted alpha matte.
        tolerance (float): Tolerance range for expected values.

    Returns:
        Dict[str, bool]: Dictionary indicating the presence of 0, 0.5, and 1.
    """
    values = {
        "background (0)": np.any(np.isclose(alpha_matte, 0, atol=tolerance)),
        "overlap (0.5)": np.any(np.isclose(alpha_matte, 0.5, atol=tolerance)),
        "foreground (1)": np.any(np.isclose(alpha_matte, 1, atol=tolerance))
    }

    return values


def _convert_binary_to_alpha_mask(binary_mask: np.ndarray, overlap_region: float = 0.05) -> np.ndarray:
    """
    Convert a binary mask to a pseudo-alpha matte for comparison.

    Args:
        binary_mask (np.ndarray): Ground truth binary mask (0 or 1).
        overlap_region (float): Fraction of the mask edges to set as overlap (0.5).

    Returns:
        np.ndarray: Alpha matte with 0 (background), 0.5 (overlap), and 1 (foreground).
    """
    from scipy.ndimage import gaussian_filter

    # Create overlap region by applying a Gaussian blur to edges
    blurred_mask = gaussian_filter(binary_mask, sigma=3)  # Smooth the edges
    alpha_mask = np.clip(blurred_mask, 0, 1)  # Ensure values are between 0 and 1

    # Set overlap region to ~0.5
    overlap_mask = (alpha_mask > overlap_region) & (alpha_mask < 1 - overlap_region)
    alpha_mask[overlap_mask] = 0.5

    return alpha_mask
