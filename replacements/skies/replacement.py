from typing import Optional

import pymatting

from pathlib import Path
import numpy as np
import cv2

from constants import OUTPUT_DIRECTORY


def replace_background(
        background_image_path: Path,
        sky_foreground: np.ndarray,
        alpha_mask: np.ndarray,
        save_image: bool = True,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    """
    Replace the sky in the original image with a new sky using alpha compositing.

    This function uses the generated sky_foreground (the refined extraction of the sky
    and any retained non-sky areas) along with a continuous alpha mask (where 1 indicates
    sky) to blend in a new sky background.

    For pixels where the alpha mask is 1, the new sky is used; where the alpha mask is 0,
    the original sky_foreground is preserved. Soft transitions are handled by the continuous mask.

    Args:
        background_image_path (Path): Path to the new sky image.
        sky_foreground (np.ndarray): The refined extraction of the sky (and retained non-sky regions)
                                     from the original image (shape: H x W x 3).
        alpha_mask (np.ndarray): Continuous alpha mask with values in [0, 1] (sky = 1).
        save_image (bool): Whether to save the blended image (default: True).
        save_dir (Path): Directory to save the blended image (default: OUTPUT_DIRECTORY).

    Returns:
        np.ndarray: The composited image where the new sky replaces the original sky.
    """
    # Ensure the new background (sky) image exists
    if not background_image_path.exists():
        raise FileNotFoundError(f"Image not found: {background_image_path}")

    # Load the new sky image
    new_sky_bgr = cv2.imread(str(background_image_path), cv2.IMREAD_COLOR)
    if new_sky_bgr is None:
        raise FileNotFoundError(f"Background image not found: {background_image_path}")

    # Convert from BGR to RGB and normalize to [0, 1]
    new_sky = cv2.cvtColor(new_sky_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    # Resize the new sky image to match the dimensions of the sky_foreground
    new_sky = cv2.resize(new_sky, dsize=(sky_foreground.shape[1], sky_foreground.shape[0]))

    # If sky_foreground has an extra alpha channel, drop it (keep only RGB channels)
    if sky_foreground.shape[2] == 4:
        sky_foreground = sky_foreground[:, :, :3]

    # Perform alpha compositing:
    #   For pixels where alpha_mask is 1 (sky), use the new sky.
    #   For pixels where alpha_mask is 0, keep the original sky_foreground.
    # With soft transitions handled by the continuous alpha_mask.
    replaced_image = (1 - alpha_mask[:, :, None]) * sky_foreground + alpha_mask[:, :, None] * new_sky

    if save_image:
        save_dir.mkdir(parents=True, exist_ok=True)
        replaced_image_path = save_dir / "replaced.png"
        pymatting.save_image(str(replaced_image_path), replaced_image)
        print(f"Replaced image saved to {replaced_image_path}")

    return replaced_image


def remove_background(
        image_path: Path,
        alpha_mask: np.ndarray,
        image_title: str,
        save_image: bool = True,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    """
    Remove the background from an image using the alpha mask.

    Args:
        image_path (Path): Path to the original image.
        alpha_mask (numpy.ndarray): Alpha mask with values between 0 and 1.
        image_title (str): Title of the image.
        save_image (bool): Whether to save the image with transparent background. Default is True.
        save_dir (Path): Directory to save the image. Default is "output".

    Returns:
        numpy.ndarray: Image with transparent background (RGBA).
    """
    # Ensure the image exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the original image using pymatting
    image = pymatting.load_image(str(image_path))

    # Ensure alpha_mask is in the correct format
    if alpha_mask.max() > 1.0:
        alpha_mask = alpha_mask / 255.0

    # Ensure the image is in float64 and has values between 0 and 1
    if image.dtype != np.float64 or image.max() > 1.0:
        image = image.astype(np.float64) / 255.0

    image_with_alpha = np.dstack((image, alpha_mask))

    if save_image:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{image_title}_removed_background.png"

        image_with_alpha_uint8 = (image_with_alpha * 255).astype(np.uint8)
        image_with_alpha_uint8 = cv2.cvtColor(image_with_alpha_uint8, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(output_path), image_with_alpha_uint8)

        print(f"Removed background image saved to {output_path}")

    return image_with_alpha
