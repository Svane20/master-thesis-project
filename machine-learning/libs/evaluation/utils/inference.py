import torch
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
from pathlib import Path
import cv2

from libs.metrics.utils import compute_evaluation_metrics, get_grad_filter
from libs.training.utils.train_utils import AverageMeter, ProgressMeter


def evaluate_model_sliding_window(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        tile_size: int,
        overlap: int,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset using sliding window inference.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
        tile_size (int): Size of the sliding window
        overlap (int): Overlap of the sliding window
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
                    alpha_pred = sliding_window_inference(
                        model=model,
                        image_tensor=inputs,
                        device=device,
                        tile_size=tile_size,
                        overlap=overlap,
                    )

                    # Expand for metric function if it expects [B, 1, H, W]
                    alpha_pred = alpha_pred.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

                    # Calculate metrics
                    scores = compute_evaluation_metrics(alpha_pred, targets, grad_filter)

                    # Accumulate metrics
                    batch_size = targets.size(0)
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
        logging.info(f"{k}: {v:.3f}")

    return metrics


def sliding_window_inference(
        model: torch.nn.Module,
        image_tensor: torch.Tensor,
        device: torch.device,
        tile_size: int,
        overlap: int,
) -> torch.Tensor:
    # Ensure we only have one image in this batch
    B, C, H, W = image_tensor.shape
    assert B == 1, f"Batch size must be 1, but got {B}."

    # Remove the batch dimension => shape [C, H, W]
    image_tensor = image_tensor[0]

    # We'll store sums of predictions here
    alpha_accum = torch.zeros((H, W), dtype=torch.float32, device=device)
    # We'll track how many predictions contributed to each pixel
    weight_map = torch.zeros((H, W), dtype=torch.float32, device=device)

    stride = tile_size - overlap
    image_tensor = image_tensor.to(device, non_blocking=True)

    # Slide over the image in a grid
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + tile_size, H)
            right  = min(left + tile_size, W)

            tile = image_tensor[:, top:bottom, left:right]
            tileH = tile.size(1)
            tileW = tile.size(2)

            # 1) Figure out how much pad is needed. E.g., pad so the tile dims are multiples of 32
            #    or at least so the net can handle.
            padH = (32 - tileH % 32) if (tileH % 32) != 0 else 0
            padW = (32 - tileW % 32) if (tileW % 32) != 0 else 0

            # 2) Pad if needed
            if padH > 0 or padW > 0:
                tile = F.pad(
                    tile,  # shape [C, tileH, tileW]
                    (0, padW, 0, padH),  # pad last 2 dims: (left_pad, right_pad, top_pad, bottom_pad)
                    mode='constant', value=0.0
                )
            # Now tile is shape [C, tileH+padH, tileW+padW]

            tile = tile.unsqueeze(0)  # [1, C, H', W']
            with torch.no_grad():
                tile_pred = model(tile)  # [1, 1, H', W']

            tile_pred = tile_pred[0, 0]  # shape [H', W']

            # 3) Crop out the pad region from tile_pred
            if padH > 0 or padW > 0:
                tile_pred = tile_pred[:tileH, :tileW]

            # 4) Accumulate
            alpha_accum[top:bottom, left:right] += tile_pred
            weight_map [top:bottom, left:right] += 1.0

    # Normalize by number of overlaps
    alpha_pred = alpha_accum / torch.clamp(weight_map, min=1e-5)
    alpha_pred = torch.clamp(alpha_pred, 0.0, 1.0)

    # shape is [H, W]
    return alpha_pred


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
        transforms (T.Compose): Transforms to apply to the input image
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
