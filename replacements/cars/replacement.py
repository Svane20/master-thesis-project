import pymatting

from pathlib import Path
import numpy as np
import cv2

from constants import OUTPUT_DIRECTORY


def replace_background(
        background_image_path: Path,
        foreground: np.ndarray,
        alpha_mask: np.ndarray,
        image_title: str,
        save_image: bool = True,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    """
    Replace the background of an image with a new background using alpha compositing.

    Args:
        background_image_path (Path): Path to the new background image.
        foreground (numpy.ndarray): Foreground image with shape (H, W, 3).
        alpha_mask (numpy.ndarray): Alpha mask with values between 0 and 1.
        image_title (str): Title of the image.
        save_image (bool): Whether to save the blended image. Default is True.
        save_dir (Path): Directory to save the blended image. Default is "output".

    Returns:
        numpy.ndarray: Blended image.
    """
    # Ensure the background image exists
    if not background_image_path.exists():
        raise FileNotFoundError(f"Image not found: {background_image_path}")

    # Load the new background image
    background_image = cv2.imread(str(background_image_path), cv2.IMREAD_COLOR)
    if background_image is None:
        raise FileNotFoundError(f"Background image not found: {background_image_path}")

    # Convert background image from BGR to RGB and normalize values to [0, 1]
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    background_image = background_image.astype(np.float64) / 255.0

    # Ensure the new background image matches the dimensions of the foreground
    background_image = cv2.resize(background_image, dsize=(foreground.shape[1], foreground.shape[0]))

    # Perform alpha compositing -> new_image = alpha * foreground + (1 - alpha) * background
    replaced_image = alpha_mask[:, :, None] * foreground + (1 - alpha_mask[:, :, None]) * background_image

    if save_image:
        save_dir.mkdir(parents=True, exist_ok=True)

        replaced_image_path = save_dir / f"{image_title}_replaced.png"
        pymatting.save_image(str(replaced_image_path), replaced_image)

        print(f"Replaced image saved to {replaced_image_path}")

    return replaced_image
