import torch
import torchvision

from pathlib import Path
import random
from PIL import Image
import numpy as np

from ..configuration.configuration import Config
from ..datasets.synthetic.transforms import get_test_transforms
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
) -> None:
    # Get the test images and masks directories
    dataset_path = Path(configuration.dataset.root) / configuration.dataset.name
    test_dir = dataset_path / "test"
    images_dir = test_dir / "images"
    masks_dir = test_dir / "masks"

    # Select a random image from the test set
    image_files = [f for f in images_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]]
    chosen_image_path = random.choice(image_files)
    image = Image.open(chosen_image_path).convert("RGB")
    image_array = np.asarray(image)

    # Get the corresponding mask based on the stem of the image path
    mask_path = masks_dir / f"{chosen_image_path.stem}.png"
    mask_rgba = Image.open(mask_path).convert("RGBA")
    mask = mask_rgba.getchannel("A")
    mask_array = np.asarray(mask)

    # Predict the mask
    raw_mask, predicted_mask, metrics = predict_image(
        image=image,
        mask=mask,
        model=model,
        transform=get_test_transforms(configuration.scratch.resolution),
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

    # Upscale the predicted mask to the original image size
    predicted_mask = np.array(
        Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize(
            (image_array.shape[1], image_array.shape[0]))).astype(
        np.float32) / 255.0

    # Save the prediction
    save_prediction(
        image=image_array,
        predicted_mask=predicted_mask,
        gt_mask=mask_array,
        metrics=metrics,
        directory=output_dir
    )

    # Get the foreground estimation
    foreground, _ = get_foreground_estimation(
        chosen_image_path,
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
