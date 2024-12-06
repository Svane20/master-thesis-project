from pathlib import Path
import numpy as np
import cv2

from constants import OUTPUT_DIRECTORY


def generate_trimap_from_alpha_mask(
        alpha_mask: np.ndarray,
        image_title: str,
        erode_iter: int = 5,
        dilate_iter: int = 5,
        save_trimap: bool = False,
        save_dir: Path = OUTPUT_DIRECTORY
) -> np.ndarray:
    # Convert alpha_mask from [0,1] to [0,255]
    binary_mask = (alpha_mask * 255).astype(np.uint8)

    # Threshold the mask to get binary regions
    _, binary_mask = cv2.threshold(binary_mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # Define a kernel for morphological operations
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    # Erode to get a definite foreground region
    foreground = cv2.erode(binary_mask, kernel, iterations=erode_iter)

    # Dilate to determine definite background region
    background = cv2.dilate(binary_mask, kernel, iterations=dilate_iter)
    background = cv2.bitwise_not(background)

    # Initialize trimap with unknown regions = 128
    trimap = np.full_like(binary_mask, fill_value=128)
    trimap[background == 255] = 0  # Background
    trimap[foreground == 255] = 255  # Foreground


    if save_trimap:
        save_dir.mkdir(parents=True, exist_ok=True)

        trimap_path = save_dir / f"{image_title}_trimap.png"
        cv2.imwrite(str(trimap_path), trimap)

        print(f"Trimap saved to {trimap_path}")

    return trimap
