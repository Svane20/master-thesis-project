import torch

from pathlib import Path
import numpy as np
import cv2

from .utils.utils import get_random_image
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
        device: torch.device,
        output_dir: Path,
        save_image: bool = True,
        seed: int = None
) -> None:
    # Get the test images and masks directories
    dataset_path = Path(configuration.dataset.root) / configuration.dataset.name
    test_dir = dataset_path / "test"
    masks_dir = test_dir / "masks"

    # Get random image from the test set
    image, chosen_image_path = get_random_image(
        configuration=configuration,
        output_dir=output_dir,
        save_image=save_image,
        seed=seed
    )

    # Get the corresponding mask based on the stem of the image path
    mask_path = masks_dir / f"{chosen_image_path.stem}.png"
    mask = cv2.imread(str(mask_path), flags=0).astype(np.float32) / 255.0

    # Predict the mask
    predicted_mask, metrics = predict_image(
        configuration=configuration,
        image=image,
        mask=mask,
        model=model,
        device=device,
        save_dir=output_dir,
        save_image=save_image,
    )

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
