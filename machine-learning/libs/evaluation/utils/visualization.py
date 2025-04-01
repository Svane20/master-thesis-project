import logging
from typing import Dict
import os
from pathlib import Path
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import matplotlib.pyplot as plt


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

    alpha_mask = predicted_mask.copy()

    # Ensure the predicted mask has the correct shape
    if alpha_mask.ndim == 3 and alpha_mask.shape[0] == 1:
        alpha_mask = alpha_mask.squeeze(0)

    # Create figure with 2 rows and 2 columns
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Display input image
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Input Image", fontsize=12)
    ax[0, 0].axis("off")

    # Display metrics
    formatted_lines = [f"{key:<15s}: {value}" for key, value in metrics.items()]
    metrics_str = "\n".join(formatted_lines)
    metrics_title = "Evaluation Summary"
    ax[0, 1].text(
        0.5, 0.5, f"{metrics_title}\n\n{metrics_str}",
        fontsize=12, ha="center", va="center",
        fontfamily="monospace",  # Better alignment
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round,pad=1")
    )
    ax[0, 1].set_title("Evaluation Metrics", fontsize=12)
    ax[0, 1].axis("off")

    # Display ground truth mask
    ax[1, 0].imshow(gt_mask, cmap="gray")
    ax[1, 0].set_title("Ground Truth Mask", fontsize=12)
    ax[1, 0].axis("off")

    # Display predicted mask
    ax[1, 1].imshow(alpha_mask, cmap="binary_r", vmin=0, vmax=1)
    ax[1, 1].set_title("Predicted Mask", fontsize=12)
    ax[1, 1].axis("off")

    # Save and show
    plt.tight_layout()
    prediction_path = directory / f"{image_name}_with_metrics.png"
    plt.savefig(prediction_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)

    logging.info(f"Prediction saved to {prediction_path}")
