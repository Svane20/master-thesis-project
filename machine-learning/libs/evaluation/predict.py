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
    # Directories
    root_directory = Path(__file__).parent.parent.parent

    # Get test transforms
    dataset_path = Path(configuration.dataset.root) / configuration.dataset.name
    transforms = get_test_transforms(configuration.scratch.resolution)

    # List all image files in the validation images folder
    images_dir = dataset_path / "test" / "images"
    image_files = list(images_dir.glob("*.png"))

    # Choose a random image file
    chosen_image_path = random.choice(image_files)
    image = Image.open(chosen_image_path).convert("RGB")
    image_array = np.asarray(image)

    # Derive the corresponding mask path by replacing "Image" with "SkyMask" in the filename
    mask_filename = chosen_image_path.stem.replace("Image", "SkyMask") + ".png"
    mask_path = dataset_path / "test" / "masks" / mask_filename
    mask_rgba = Image.open(mask_path).convert("RGBA")
    alpha_channel = mask_rgba.getchannel("A")
    mask = alpha_channel
    mask_array = np.asarray(mask)

    # Predict the mask
    raw_mask, predicted_mask, metrics = predict_image(
        image=image,
        mask=mask,
        model=model,
        transform=transforms,
        device=device
    )

    # Save the predicted mask
    torchvision.utils.save_image(raw_mask, f"{output_dir}/output.png")

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
    new_sky_path = root_directory / "libs" / "replacements" / "skies" / "new_sky.webp"
    replace_background(
        new_sky_path,
        foreground,
        predicted_mask,
        save_dir=output_dir,
        save_image=True,
    )
