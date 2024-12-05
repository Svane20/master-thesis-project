import pymatting

from pathlib import Path
import numpy as np
from typing import Tuple
import cv2

from constants import OUTPUT_DIRECTORY


def generate_smooth_alpha_mask_from_binary_mask(
        image_path: Path,
        image_title: str,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: Tuple[float, float] = (0, 0),
        save_alpha: bool = True,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    """
    Generate a smooth alpha mask from a binary mask.

    Parameters:
        image_path (Path): Path to the binary mask.
        image_title (str): Title of the image.
        blur_kernel_size (tuple): Kernel size for Gaussian blur.
        blur_sigma (tuple): Sigma values for Gaussian blur in x and y directions.
        save_alpha (bool): Whether to save the alpha mask to disk. Default is True.
        save_dir (Path): Directory to save the alpha mask. Default is "output".

    Returns:
        numpy.ndarray: Alpha mask.
    """
    # Check if the binary mask exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load binary mask
    binary_mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if binary_mask is None:
        raise FileNotFoundError(f"Binary mask not found: {image_path}")

    _, binary_mask = cv2.threshold(binary_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    blurred_mask = cv2.GaussianBlur(binary_mask, ksize=blur_kernel_size, sigmaX=blur_sigma[0], sigmaY=blur_sigma[1])

    alpha_mask = blurred_mask / 255.0

    if save_alpha:
        save_dir.mkdir(parents=True, exist_ok=True)

        alpha_path = save_dir / f"{image_title}_alpha.png"
        cv2.imwrite(str(alpha_path), (alpha_mask * 255).astype("uint8"))

        print(f"Alpha mask saved to {alpha_path}")

    return alpha_mask


def generate_alpha_mask_from_trimap(
        image: np.ndarray,
        trimap: np.ndarray,
        image_title: str,
        save_alpha: bool = True,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    """
    Generate an alpha mask from an image and a trimap using PyMatting.

    Parameters:
        image (numpy.ndarray): Image with shape (H, W, 3).
        trimap (numpy.ndarray): Trimap with values 0 (background), 128 (unknown), and 255 (foreground).
        image_title (str): Title of the image.
        save_alpha (bool): Whether to save the alpha mask to disk. Default is True.
        save_dir (Path): Directory to save the alpha mask. Default is "output".

    Returns:
        numpy.ndarray: Alpha mask.
    """
    if image.shape[:2] != trimap.shape[:2]:
        raise ValueError(f"Image and trimap dimensions do not match: {image.shape}, {trimap.shape}")

    alpha = pymatting.estimate_alpha_cf(
        image,
        trimap,
        laplacian_kwargs={"epsilon": 1e-6},
        cg_kwargs={"maxiter": 2000}
    )

    if save_alpha:
        save_dir.mkdir(parents=True, exist_ok=True)

        alpha_path = save_dir / f"{image_title}_alpha.png"
        pymatting.save_image(str(alpha_path), alpha)

        print(f"Alpha mask saved to {alpha_path}")

    return alpha
