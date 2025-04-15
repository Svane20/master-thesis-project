from pathlib import Path
import numpy as np
import cv2
import random
from PIL import Image
from typing import Tuple

from libs.replacement.foreground_estimation import do_foreground_estimation


def do_sky_replacement(image: Image, alpha_mask: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform sky replacement on the input image using the provided alpha mask.

    Args:
        image (Image): The original image (PIL Image).
        alpha_mask (np.ndarray): Continuous alpha mask with values in [0, 1] (sky = 1).
        target_size (int): Target size of the output image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The composited image where the new sky replaces the original sky.
            - The foreground image (refined extraction of the sky and any retained non-sky areas).
    """
    # Resize the image to the target size
    image = image.resize((target_size, target_size), Image.LANCZOS)

    # Convert the image to a NumPy array (H x W x 3) and normalize to [0, 1]
    image_np = np.array(image) / 255.0

    # Perform foreground estimation
    foreground = do_foreground_estimation(image_np, alpha_mask)

    # Replace the background using the alpha mask
    return _replace_background(foreground, alpha_mask), foreground


def _replace_background(foreground: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
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
    new_sky_path = Path(__file__).parent.parent / "backgrounds/skies/francesco-ungaro-i75WTJn-RBY-unsplash.jpg"
    if not new_sky_path.exists():
        raise FileNotFoundError(f"Image not found: {new_sky_path}")

    # Load the new sky image (BGR) and convert to RGB; normalize to [0, 1]
    new_sky_bgr = cv2.imread(str(new_sky_path), cv2.IMREAD_COLOR)
    if new_sky_bgr is None:
        raise FileNotFoundError(f"Background image not found: {new_sky_path}")
    new_sky = cv2.cvtColor(new_sky_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Obtain target dimensions from the foreground image.
    h, w = foreground.shape[:2]

    # Get the current dimensions (height, width) of the sky image.
    sky_h, sky_w = new_sky.shape[:2]

    # If the sky image is smaller than the foreground in any dimension,
    # scale it up using a scaling factor (with Lanczos interpolation).
    if sky_w < w or sky_h < h:
        scale = max(w / sky_w, h / sky_h)
        new_size = (int(sky_w * scale), int(sky_h * scale))  # (width, height)
        new_sky = cv2.resize(new_sky, new_size, interpolation=cv2.INTER_LANCZOS4)
        sky_h, sky_w = new_sky.shape[:2]

    # Determine the maximum valid top-left coordinates for cropping.
    max_left = sky_w - w
    max_top = sky_h - h
    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0

    # Crop the sky image to the target size using the random offsets.
    new_sky = new_sky[top:top + h, left:left + w]

    # Convert the foreground to float32 and normalize to [0, 1] if necessary.
    if foreground.dtype != np.float32:
        foreground = foreground.astype(np.float32) / 255.0
    # Remove any extra alpha channel if present.
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]

    # Ensure that the alpha mask values are within [0, 1].
    alpha_mask = np.clip(alpha_mask, 0, 1)

    # Perform alpha compositing:
    #   For pixels where alpha_mask is 1, use the new sky;
    #   where it is 0, keep the foreground.
    return (1 - alpha_mask[..., None]) * foreground + alpha_mask[..., None] * new_sky
