import pymatting

from typing import Tuple
from pathlib import Path
import numpy as np
import logging

from constants import OUTPUT_DIRECTORY


def get_foreground_estimation(
        image_path: Path,
        alpha_mask: np.ndarray,
        save_foreground: bool = False,
        save_background: bool = False,
        save_dir: Path = OUTPUT_DIRECTORY
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the foreground and background of an image.

    Args:
        image_path (Path): Path to the image.
        alpha_mask (numpy.ndarray): Alpha mask with values between 0 and 1.
        save_foreground (bool): Whether to save the foreground image. Default is True.
        save_background (bool): Whether to save the background image. Default is True.
        save_dir (Path): Directory to save the foreground and background images. Default is "output".

    Returns:
        numpy.ndarray: Foreground image.
        numpy.ndarray: Background image.

    """
    # Check if the image exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the image
    image = pymatting.load_image(str(image_path))

    # IMPORTANT: In our training data, the alpha matte is generated so that the sky/HDRI (our target subject)
    # is marked as 1 and the foreground/geometry is 0. However, the pymatting.estimate_foreground_ml function
    # is designed to extract the "foreground" (the object of interest) as the opaque region (alpha = 1).
    #
    # In our sky replacement pipeline, we want to preserve the geometry (foreground) from the original image
    # and replace the sky (background) with a new sky. To achieve this using pymatting, we need to extract
    # the geometry as the foreground. Therefore, we invert the alpha mask (using 1 - alpha_mask) so that:
    #    - The original geometry (which was 0) becomes 1 (treated as the foreground),
    #    - The sky (which was 1) becomes 0 (treated as background).
    #
    # This inversion aligns the alpha mask with pymatting’s expectation for foreground extraction.
    inverted_alpha_mask = 1 - alpha_mask

    print(f"Shape of alpha mask: {alpha_mask.shape}")
    print(f"Shape of image: {image.shape}")

    # Estimate the foreground and background
    foreground, background = pymatting.estimate_foreground_ml(image=image, alpha=inverted_alpha_mask, return_background=True)

    if save_foreground:
        save_dir.mkdir(parents=True, exist_ok=True)

        foreground_path = save_dir / "foreground.png"
        pymatting.save_image(str(foreground_path), image=foreground)

        logging.info(f"Foreground saved to {foreground_path}")

    if save_background:
        save_dir.mkdir(parents=True, exist_ok=True)

        background_path = save_dir / "background.png"
        pymatting.save_image(str(background_path), image=background)

        logging.info(f"Background saved to {background_path}")

    return foreground, background
