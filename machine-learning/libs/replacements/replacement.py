import pymatting

from pathlib import Path
import numpy as np
import cv2
import logging


def replace_background(
        foreground: np.ndarray,
        alpha_mask: np.ndarray,
        save_dir: Path,
        save_image: bool = True,
) -> np.ndarray:
    """
    Replace the sky in the original image with a new sky using alpha compositing.

    This function uses the generated sky_foreground (the refined extraction of the sky
    and any retained non-sky areas) along with a continuous alpha mask (where 1 indicates
    sky) to blend in a new sky background.

    For pixels where the alpha mask is 1, the new sky is used; where the alpha mask is 0,
    the original sky_foreground is preserved. Soft transitions are handled by the continuous mask.

    Args:
        foreground (np.ndarray): The refined extraction of the sky (and retained non-sky regions)
                                     from the original image (shape: H x W x 3).
        alpha_mask (np.ndarray): Continuous alpha mask with values in [0, 1] (sky = 1).
        save_image (bool): Whether to save the blended image (default: True).
        save_dir (Path): Directory to save the blended image (default: OUTPUT_DIRECTORY).

    Returns:
        np.ndarray: The composited image where the new sky replaces the original sky.
    """
    root_directory = Path(__file__).parent.parent.parent
    background_image_path = root_directory / "libs" / "replacements" / "skies" / "new_sky.webp"

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
    new_sky = cv2.resize(new_sky, dsize=(foreground.shape[1], foreground.shape[0]))

    # If sky_foreground has an extra alpha channel, drop it (keep only RGB channels)
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]

    # Perform alpha compositing:
    #   For pixels where alpha_mask is 1 (sky), use the new sky.
    #   For pixels where alpha_mask is 0, keep the original sky_foreground.
    # With soft transitions handled by the continuous alpha_mask.
    replaced_image = (1 - alpha_mask[:, :, None]) * foreground + alpha_mask[:, :, None] * new_sky

    if save_image:
        save_dir.mkdir(parents=True, exist_ok=True)
        replaced_image_path = save_dir / "replaced.png"
        pymatting.save_image(str(replaced_image_path), replaced_image)
        logging.info(f"Replaced image saved to {replaced_image_path}")

    return replaced_image
