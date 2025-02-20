import pymatting

import cv2
from pathlib import Path
import numpy as np
import random
import logging

from constants import DATA_BACKGROUNDS_DIRECTORY, OUTPUT_DIRECTORY
from skies.configuration.base import get_configurations, Configuration
from skies.custom_logging.custom_logger import setup_logging
from skies.foreground_estimation import get_foreground_estimation
from skies.replacement import replace_background


def get_random_image_and_alpha_matte(config: Configuration):
    # Directories
    source_directory = Path(config.source_directory)
    images_directory = source_directory / "images"
    masks_directory = source_directory / "masks"

    # List all image files
    image_files = list(images_directory.glob("*.png"))
    if not image_files:
        logging.error("No image files found in directory: %s", images_directory)
        raise FileNotFoundError(f"No image files found in directory: {images_directory}")

    # Choose a random image
    random_image = random.choice(image_files)
    logging.info("Chosen image: %s", random_image)

    # Derive the corresponding mask file name by replacing '_Image_' with '_SkyMask_'
    mask_filename = random_image.name.replace("_Image_", "_SkyMask_")
    mask_path = masks_directory / mask_filename

    if not mask_path.exists():
        logging.error("Corresponding mask file not found for image %s: expected %s", random_image, mask_path)
        raise FileNotFoundError(f"Corresponding mask file not found for image {random_image}: expected {mask_path}")

    return random_image, mask_path


if __name__ == "__main__":
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Set the seed for reproducibility
    if configuration.seed is not None:
        random.seed(configuration.seed)
        np.random.seed(configuration.seed)

    # Get a random image and its corresponding alpha matte
    image_path, alpha_mask_path = get_random_image_and_alpha_matte(configuration)

    # Define the file paths for the new sky background
    new_sky_path = DATA_BACKGROUNDS_DIRECTORY / "new_sky.webp"

    # Ensure the files exist
    if not new_sky_path.exists():
        raise FileNotFoundError(f"New sky background not found: {new_sky_path}")

    # Load the image and convert it to RGB
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    height, width = image.shape[:2]

    # Load the alpha matting mask
    alpha_image = cv2.imread(str(alpha_mask_path), cv2.IMREAD_UNCHANGED)
    if alpha_image is None:
        raise FileNotFoundError(f"Alpha mask not found: {alpha_mask_path}")
    if alpha_image.shape[2] == 4:
        # The alpha channel is the 4th channel (index 3).
        alpha_mask = alpha_image[..., 3].astype(np.float64) / 255.0
    else:
        # Fallback: if the image is not RGBA, assume it's grayscale.
        alpha_mask = alpha_image.astype(np.float64) / 255.0

    # Save the original image with the alpha mask
    save_dir = OUTPUT_DIRECTORY

    pymatting.save_image(str(save_dir / "_image.png"), image=image_bgr)
    pymatting.save_image(str(save_dir / "alpha_matte.png"), image=alpha_image)


    # Perform foreground estimation to get the foreground and background images
    foreground, _ = get_foreground_estimation(
        image_path,
        alpha_mask,
        save_foreground=True,
    )

    # Load the new sky image
    new_sky_bgr = cv2.imread(str(new_sky_path), cv2.IMREAD_COLOR)
    if new_sky_bgr is None:
        raise FileNotFoundError(f"New sky background not found: {new_sky_path}")
    new_sky = cv2.cvtColor(new_sky_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    new_sky = cv2.resize(new_sky, (width, height))

    # Perform sky replacement
    replace_background(
        new_sky_path,
        foreground,
        alpha_mask,
        save_image=True,
        save_dir=OUTPUT_DIRECTORY
    )
