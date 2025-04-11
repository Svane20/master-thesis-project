import torch
import torchvision.transforms as T

import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
from pathlib import Path
import cv2

from libs.metrics.utils import compute_evaluation_metrics, get_grad_filter
from libs.training.utils.train_utils import AverageMeter, ProgressMeter, MemMeter


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
    """
    # Set model to evaluation mode
    model.eval()

    # Gradient filter
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    grad_filter = get_grad_filter(device=device, dtype=dtype)

    # Stat meters
    mem_meter = MemMeter(name="Mem (GB)", device=str(device), fmt=":.2f")

    # Metric meters
    mae_meter = AverageMeter(name="MAE", device=str(device), fmt=":.2e")
    mse_meter = AverageMeter(name="MSE", device=str(device), fmt=":.2e")
    sad_meter = AverageMeter(name="SAD", device=str(device), fmt=":.2e")
    grad_meter = AverageMeter(name="Grad", device=str(device), fmt=":.2e")
    conn_meter = AverageMeter(name="Conn", device=str(device), fmt=":.2e")

    # Progress bar
    progress = ProgressMeter(
        num_batches=len(data_loader),
        meters=[
            mae_meter,
            mse_meter,
            sad_meter,
            grad_meter,
            conn_meter,
            mem_meter,
        ],
        real_meters={},
        prefix="Test | Epoch: [{}]".format(1),
    )

    for batch_idx, sample in enumerate(data_loader):
        # Move data to device
        inputs = sample["image"].to(device, non_blocking=True)
        targets = sample["alpha"].to(device, non_blocking=True)

        try:
            with torch.no_grad():
                with torch.amp.autocast(
                        device_type=device.type,
                        enabled=torch.cuda.is_available(),
                        dtype=dtype
                ):
                    # Get predictions
                    outputs = model(inputs)

                    # Calculate metrics
                    scores = compute_evaluation_metrics(outputs, targets, grad_filter)

                    # Accumulate metrics
                    batch_size = targets.size(0)
                    mse_meter.update(scores["mse"], n=batch_size)
                    mae_meter.update(scores["mae"], n=batch_size)
                    sad_meter.update(scores["sad"], n=batch_size)
                    grad_meter.update(scores["grad"], n=batch_size)
                    conn_meter.update(scores["conn"], n=batch_size)

            # Measure memory usage
            if torch.cuda.is_available():
                mem_meter.update(reset_peak_usage=True)

            # Update the progress bar every 10 batches
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e

    # Logs metrics
    metrics = {
        "mae": mae_meter.avg,
        "mse": mse_meter.avg,
        "grad": grad_meter.avg,
        "sad": sad_meter.avg,
        "conn": conn_meter.avg,
    }

    logging.info("Evaluation finished.")
    for k, v in metrics.items():
        logging.info(f"{k}: {v:.3f}")

    return metrics


def predict(
        model: torch.nn.Module,
        transforms: T.Compose,
        image: np.ndarray,
        device: torch.device,
        save_dir: Path,
        save_image: bool = True,
) -> np.ndarray:
    """
    Predict the alpha matte for an image

    Args:
        model (torch.nn.Module): Model to use for prediction
        transforms (T.Compose): Transforms to apply to the input
        image (np.ndarray): Input image
        device (torch.device): Device to use for evaluation
        save_dir (Path): Directory to save the predicted image
        save_image (bool): Whether to save the predicted image

    Returns:
        np.ndarray: Predicted alpha matte
    """
    # Apply the transforms and move the image to the device
    image_tensor = transforms({"image": image})["image"]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    enabled = torch.cuda.is_available()

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
            outputs = model(image_tensor)

    # Convert the output to a numpy array and remove the batch dimension
    predicted_alpha = outputs.detach().cpu().numpy()
    predicted_alpha = np.squeeze(predicted_alpha)

    if save_image:
        logging.info(f"Saved predicted image to {save_dir / 'predicted.png'}")
        cv2.imwrite(str(save_dir / "predicted.png"), (predicted_alpha * 255).astype("uint8"))

    return predicted_alpha


def predict_image(
        transforms: T.Compose,
        image: Image,
        mask: Image,
        model: torch.nn.Module,
        device: torch.device,
        save_dir: Path,
        save_image: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Predict the alpha matte for an image.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Ground truth mask.
        model (torch.nn.Module): Model to use for prediction.
        device (torch.device): Device to use for evaluation.
        save_dir (Path): Directory to save the predicted image.
        save_image (bool): Whether to save the predicted image.

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: Predicted mask and evaluation metrics.
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
    sample = transforms({"image": image, "alpha": mask})
    image_tensor, mask_tensor = sample["image"], sample["alpha"]

    # Move tensors to device and add batch dimension
    image_tensor, mask_tensor = image_tensor.unsqueeze(0).to(device), mask_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=dtype):
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
        logging.info(f"{k}: {v:.3f}")

    # Convert alpha matte to numpy array
    pred_alpha = outputs.detach().cpu().numpy()
    pred_alpha = np.squeeze(pred_alpha)

    if save_image:
        logging.info(f"Saved predicted image to {save_dir / 'predicted.png'}")
        cv2.imwrite(str(save_dir / "predicted.png"), (pred_alpha * 255).astype("uint8"))

    return pred_alpha, metrics
