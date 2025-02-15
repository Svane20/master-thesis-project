import cv2
import numpy as np

from constants import DATA_IMAGES_DIRECTORY, DATA_MASKS_DIRECTORY, DATA_BACKGROUNDS_DIRECTORY, OUTPUT_DIRECTORY
from skies.foreground_estimation import get_foreground_estimation
from skies.replacement import replace_background

if __name__ == "__main__":
    # Define the file paths
    image_path = DATA_IMAGES_DIRECTORY / "Image_2.png"
    alpha_mask_path = DATA_MASKS_DIRECTORY / "HDRIMask_2.png"
    new_sky_path = DATA_BACKGROUNDS_DIRECTORY / "new_sky.webp"

    # Ensure the files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not alpha_mask_path.exists():
        raise FileNotFoundError(f"Alpha mask not found: {alpha_mask_path}")
    if not new_sky_path.exists():
        raise FileNotFoundError(f"New sky background not found: {new_sky_path}")

    # Load the image and convert it to RGB
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
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
