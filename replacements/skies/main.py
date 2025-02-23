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


def is_valid_alpha_matte(mask_path: Path) -> bool:
    """
    Check if the mask at mask_path is a valid continuous alpha matte.
    For our purposes, a valid matte should have an alpha channel with values normalized between 0 and 1,
    and it should span a significant range (e.g., minimum below 0.1 and maximum above 0.9).
    """
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        logging.error("Could not load mask: %s", mask_path)
        return False

    # Verify the mask has 4 channels (RGBA).
    if len(mask_image.shape) < 3 or mask_image.shape[2] != 4:
        logging.error("Mask does not have 4 channels: %s", mask_path)
        return False

    # Extract the alpha channel (the 4th channel).
    alpha_channel = mask_image[..., 3].astype(np.float64) / 255.0

    # Get the minimum and maximum alpha values.
    min_alpha = np.min(alpha_channel)
    max_alpha = np.max(alpha_channel)

    # Check if the alpha matte spans a significant range.
    if min_alpha < 0.1 and max_alpha > 0.9:
        return True
    else:
        logging.warning("Alpha matte does not span full range: min %f, max %f", min_alpha, max_alpha)
        return False


def get_random_image_and_alpha_matte(config: Configuration):
    # Start at the source directory.
    source_directory = Path(config.source_directory)

    # Find datetime folders (subdirectories) inside the source directory.
    datetime_folders = [d for d in source_directory.iterdir() if d.is_dir()]
    if not datetime_folders:
        logging.error("No datetime folders found in source directory: %s", source_directory)
        raise FileNotFoundError(f"No datetime folders found in source directory: {source_directory}")

    # For this example, we'll just choose the first (or you could sort or randomly pick).
    chosen_datetime_folder = random.choice(datetime_folders)
    logging.info("Chosen datetime folder: %s", chosen_datetime_folder)

    # Now build the paths for images and masks inside the chosen datetime folder.
    images_directory = chosen_datetime_folder / "images"
    masks_directory = chosen_datetime_folder / "masks"

    # List all image files in the images directory.
    image_files = list(images_directory.glob("*.png"))
    if not image_files:
        logging.error("No image files found in directory: %s", images_directory)
        raise FileNotFoundError(f"No image files found in directory: {images_directory}")

    valid_pairs = []
    for image_file in image_files:
        # Derive the corresponding mask file name by replacing '_Image_' with '_SkyMask_'
        mask_filename = image_file.name.replace("Image_", "SkyMask_")
        mask_path = masks_directory / mask_filename
        if mask_path.exists() and is_valid_alpha_matte(mask_path):
            valid_pairs.append((image_file, mask_path))
        else:
            logging.warning("Skipping %s because its corresponding mask is missing or invalid.", image_file)

    if not valid_pairs:
        raise FileNotFoundError("No valid image and alpha matte pairs found in the source directory.")

    chosen_pair = random.choice(valid_pairs)

    logging.info(f"Chosen image: {chosen_pair[0]}")
    logging.info(f"Chosen mask: {chosen_pair[1]}")

    # Save the chosen image and mask
    chosen_image_path = OUTPUT_DIRECTORY / "chosen_image.png"
    pymatting.save_image(str(chosen_image_path), image=cv2.imread(str(chosen_pair[0]), cv2.IMREAD_UNCHANGED))
    logging.info(f"Image saved to {chosen_image_path}")

    chosen_mask_path = OUTPUT_DIRECTORY / "chosen_mask.png"
    pymatting.save_image(str(chosen_mask_path), image=cv2.imread(str(chosen_pair[1]), cv2.IMREAD_UNCHANGED))
    logging.info(f"Mask saved to {chosen_mask_path}")

    return chosen_pair


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
