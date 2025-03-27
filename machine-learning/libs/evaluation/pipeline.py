import torch

from pathlib import Path
import cv2
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import matplotlib.pyplot as plt

from ..configuration.configuration import Config
from ..replacements.foreground_estimation import get_foreground_estimation
from ..replacements.replacement import replace_background
from ..training.utils.logger import setup_logging
from .utils.inference import predict
from .utils.utils import get_random_image

setup_logging(__name__)


def run_pipeline(
        configuration: Config,
        model: torch.nn.Module,
        device: torch.device,
        output_dir: Path,
        save_image: bool = True,
        seed: int = None
) -> None:
    # Get random image from the test set
    image, chosen_image_path = get_random_image(
        configuration=configuration,
        output_dir=output_dir,
        save_image=save_image,
        seed=seed
    )

    # Perform inference
    predicted_alpha = predict(
        configuration=configuration,
        model=model,
        device=device,
        image=image,
        save_dir=output_dir,
        save_image=save_image,
    )

    # Downscale the image to fit the predicted alpha
    h, w = predicted_alpha.shape
    downscaled_image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)

    # Perform foreground estimation
    foreground, _ = get_foreground_estimation(
        image=downscaled_image,
        alpha_mask=predicted_alpha,
        save_dir=output_dir,
        save_foreground=True
    )

    # Replace the background with the new sky
    replaced_image = replace_background(
        foreground=foreground,
        alpha_mask=predicted_alpha,
        save_dir=output_dir,
        save_image=True
    )

    # Visualize the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Row 1: Original Image and Predicted Alpha
    axes[0, 0].imshow(cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(predicted_alpha, cmap="gray")
    axes[0, 1].set_title("Predicted Alpha")
    axes[0, 1].axis("off")

    # Row 2: Foreground and Replaced background image
    axes[1, 0].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Estimated Foreground")
    axes[1, 0].axis("off")

    replaced_image_uint8 = (np.clip(replaced_image, 0, 1) * 255).astype(np.uint8)
    axes[1, 1].imshow(replaced_image_uint8)
    axes[1, 1].set_title("Replaced Background")
    axes[1, 1].axis("off")

    plt.suptitle(f"Image: {chosen_image_path.name}", fontsize=16)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / "summary_plot.png")
    plt.show()
    plt.close()
