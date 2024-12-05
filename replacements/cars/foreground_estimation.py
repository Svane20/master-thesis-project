import pymatting

from typing import Tuple
from pathlib import Path
import numpy as np
import cv2

from constants import OUTPUT_DIRECTORY


def get_foreground_estimation(
        image_path: Path,
        alpha_mask: np.ndarray,
        image_title: str,
        save_foreground: bool = False,
        save_background: bool = False,
        save_dir: Path = OUTPUT_DIRECTORY
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the foreground and background of an image.

    Args:
        image_path (Path): Path to the image.
        alpha_mask (numpy.ndarray): Alpha mask with values between 0 and 1.
        image_title (str): Title of the image.
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

    # Estimate the foreground and background
    foreground, background = pymatting.estimate_foreground_ml(image=image, alpha=alpha_mask, return_background=True)

    if save_foreground:
        save_dir.mkdir(parents=True, exist_ok=True)

        foreground_path = save_dir / f"{image_title}_foreground.png"
        pymatting.save_image(str(foreground_path), image=foreground)

        print(f"Foreground saved to {foreground_path}")

    if save_background:
        save_dir.mkdir(parents=True, exist_ok=True)

        background_path = save_dir / f"{image_title}_background.png"
        pymatting.save_image(str(background_path), image=background)

        print(f"Background saved to {background_path}")

    return foreground, background
