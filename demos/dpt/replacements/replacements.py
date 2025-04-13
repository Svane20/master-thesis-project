from pathlib import Path
from PIL import Image
import numpy as np


def sky_replacement(foreground, alpha_mask):
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
