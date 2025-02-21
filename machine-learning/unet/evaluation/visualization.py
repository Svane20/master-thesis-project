from typing import Dict

from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def resize_to_match(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize an image or mask to match the target shape.

    Args:
        image (np.ndarray): Input image or mask.
        target_shape (tuple): Desired shape (height, width).

    Returns:
        np.ndarray: Resized image or mask.
    """
    pil_image = Image.fromarray((image * 255).astype(np.uint8) if image.dtype != np.uint8 else image)
    resized_pil = pil_image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(resized_pil, dtype=np.float32) / 255.0 if image.dtype != np.uint8 else np.array(resized_pil)


def save_prediction(
        image: np.ndarray,
        predicted_mask: np.ndarray,
        gt_mask: np.ndarray,
        metrics: Dict[str, float],
        directory: Path,
        image_name: str = "prediction"
) -> None:
    """
    Save the input image, ground truth mask, predicted mask, and metrics.

    Args:
        image (np.ndarray): Input image.
        predicted_mask (np.ndarray): Predicted mask.
        gt_mask (np.ndarray): Ground truth mask.
        metrics (Dict[str, float]): Evaluation metrics.
        directory (Path): Directory to save the visualization.
        image_name (str): Name of the image for saving and identification.
    """
    # Create the directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)

    # Ensure the predicted mask has the correct shape
    if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)

    # Resize all images to match the predicted mask dimensions
    target_shape = predicted_mask.shape
    image_resized = resize_to_match(image, target_shape)
    gt_mask_resized = resize_to_match(gt_mask, target_shape)

    # Create figure with 2 rows and 2 columns
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Display input image
    ax[0, 0].imshow(image_resized)
    ax[0, 0].set_title("Input Image", fontsize=12)
    ax[0, 0].axis("off")

    # Display metrics
    metrics_str = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    ax[0, 1].text(
        0.5, 0.5, metrics_str, fontsize=14, ha="center", va="center",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
    )
    ax[0, 1].set_title("Evaluation Metrics", fontsize=12)
    ax[0, 1].axis("off")

    # Display ground truth mask
    gt_mask_stretched = (gt_mask_resized - 0.93) / (1 - 0.93)
    gt_mask_stretched = np.clip(gt_mask_stretched, 0, 1)
    ax[1, 0].imshow(gt_mask_stretched, cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title("Ground Truth Mask", fontsize=12)
    ax[1, 0].axis("off")

    # Display predicted mask
    ax[1, 1].imshow(predicted_mask, cmap="binary_r", vmin=0, vmax=1)
    ax[1, 1].set_title("Predicted Mask", fontsize=12)
    ax[1, 1].axis("off")

    # Save and show
    plt.tight_layout()
    prediction_path = directory / f"{image_name}_with_metrics.png"
    plt.savefig(prediction_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Prediction saved to {prediction_path}")
