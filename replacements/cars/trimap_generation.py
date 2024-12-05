import pymatting

from pathlib import Path
import numpy as np
import cv2

from constants import OUTPUT_DIRECTORY


def generate_trimap_from_binary_mask(
        image_path: Path,
        image_title: str,
        kernel_size: int = 5,
        erosion_iter: int = 5,
        dilation_iter: int = 5,
        save_trimap: bool = True,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    """
    Generate a trimap from a binary mask.

    Args:
        image_path (Path): Path to the binary mask.
        image_title (str): Title of the image.
        kernel_size (int): Size of the kernel used for dilation and erosion.
        erosion_iter (int): Number of erosion iterations for definite foreground.
        dilation_iter (int): Number of dilation iterations for unknown regions.
        save_trimap (bool): Whether to save the trimap to disk. Default is True.
        save_dir (Path): Directory to save the trimap. Default is "output".

    Returns:
        numpy.ndarray: Trimap with values normalized to 0 (background), 0.5 (unknown), and 1 (foreground).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    binary_mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if binary_mask is None:
        raise FileNotFoundError(f"Binary mask not found: {image_path}")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Erode the binary mask to define definite foreground
    foreground = cv2.erode(binary_mask, kernel, iterations=erosion_iter)
    background = cv2.dilate(binary_mask, kernel, iterations=dilation_iter)
    unknown = cv2.subtract(background, foreground)

    # Create trimap
    trimap = np.zeros_like(binary_mask, dtype=np.float32)
    trimap[unknown > 0] = 0.5  # Unknown region
    trimap[foreground > 0] = 1.0  # Foreground
    trimap[binary_mask == 0] = 0.0  # Background

    if save_trimap:
        save_dir.mkdir(parents=True, exist_ok=True)

        trimap_path = save_dir / f"{image_title}_trimap.png"
        pymatting.save_image(str(trimap_path), trimap)

        print(f"Trimap saved to {trimap_path}")

    return trimap
