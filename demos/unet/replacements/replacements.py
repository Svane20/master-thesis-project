from pathlib import Path
from PIL import Image
import numpy as np


def sky_replacement(foreground: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
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

    Returns:
        np.ndarray: The composited image where the new sky replaces the original sky.
    """
    # Load the new sky image
    current_directory = Path(__file__).parent.parent
    new_sky_path = current_directory / "assets/skies/new_sky.webp"
    new_sky_img = Image.open(new_sky_path).convert("RGB")

    # Resize to match foreground dimensions
    h, w = foreground.shape[:2]
    new_sky_img = new_sky_img.resize(size=(w, h), resample=Image.Resampling.LANCZOS)

    # Convert new_sky and foreground to float32 numpy arrays in range [0, 1]
    new_sky = np.asarray(new_sky_img).astype(np.float32) / 255.0
    if foreground.dtype != np.float32:
        foreground = foreground.astype(np.float32) / 255.0

    # If foreground has an alpha channel, drop it
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]

    # Ensure values are in [0, 1]
    alpha_mask = np.clip(alpha_mask, a_min=0, a_max=1)

    return (1 - alpha_mask[:, :, None]) * foreground + alpha_mask[:, :, None] * new_sky
