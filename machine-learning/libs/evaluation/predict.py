import logging

import torch
import torchvision

from pathlib import Path
import random
import numpy as np
import cv2

from ..configuration.configuration import Config
from ..replacements.foreground_estimation import get_foreground_estimation
from ..replacements.replacement import replace_background
from ..training.utils.logger import setup_logging

from .utils.inference import predict_image
from .utils.visualization import save_prediction

setup_logging(__name__)


def run_prediction(
        configuration: Config,
        model: torch.nn.Module,
        transforms: torchvision.transforms.Compose,
        device: torch.device,
        output_dir: Path,
) -> None:
    # Get the test images and masks directories
    dataset_path = Path(configuration.dataset.root) / configuration.dataset.name
    test_dir = dataset_path / "test"
    images_dir = test_dir / "images"
    masks_dir = test_dir / "masks"

    # Select a random image from the test set
    image_files = [f for f in images_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]]
    chosen_image_path = random.choice(image_files)
    logging.info(f"Chosen image: {chosen_image_path}")
    image = cv2.imread(str(chosen_image_path))
    cv2.imwrite(str(output_dir / f"input{chosen_image_path.suffix}"), image)

    # Get the corresponding mask based on the stem of the image path
    mask_path = masks_dir / f"{chosen_image_path.stem}.png"
    mask = cv2.imread(str(mask_path), flags=0).astype(np.float32) / 255.0

    # Predict the mask
    raw_mask, predicted_mask, metrics = predict_image(
        image=image,
        mask=mask,
        model=model,
        transform=transforms,
        device=device
    )

    # Save the predicted mask
    torchvision.utils.save_image(raw_mask, output_dir / "output.png")

    # Normalize the alpha mask to [0, 1]
    if predicted_mask.shape[2] == 4:
        # Extract the alpha channel properly.
        predicted_mask = predicted_mask[..., 3].astype(np.float64) / 255.0
    else:
        # If there’s no fourth channel, assume the mask is the output.
        predicted_mask = np.squeeze(predicted_mask, axis=0)

    # Downscale the image to fit the predicted alpha
    h, w = predicted_mask.shape
    downscaled_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    # Save the prediction
    save_prediction(
        image=downscaled_image,
        predicted_mask=predicted_mask,
        gt_mask=mask,
        metrics=metrics,
        directory=output_dir
    )

    # Get the foreground estimation
    foreground, _ = get_foreground_estimation(
        image=downscaled_image,
        alpha_mask=predicted_mask,
        save_dir=output_dir,
        save_foreground=True,
    )

    # Replace the background with a new sky
    replace_background(
        foreground=foreground,
        alpha_mask=predicted_mask,
        save_dir=output_dir,
        save_image=True,
    )
