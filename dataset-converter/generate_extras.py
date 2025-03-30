from pathlib import Path
import logging
import shutil
from tqdm.auto import tqdm
import cv2
import numpy as np

from configuration.root import get_configurations
from custom_logging.custom_logger import setup_logging
from utils import collect_samples


def generate_extras(destination_directory: Path) -> None:
    """
    Process the data from the source directory and save the results to the destination directory.

    Args:
        destination_directory (Path): The destination directory to save the processed data.
    """
    # Directories
    train_dest = destination_directory / "train"
    trimaps_dest = train_dest / "trimaps"
    fg_dest = train_dest / "fg"
    bg_dest = train_dest / "bg"

    # Create directories
    trimaps_dest.mkdir(parents=True, exist_ok=True)
    fg_dest.mkdir(parents=True, exist_ok=True)
    bg_dest.mkdir(parents=True, exist_ok=True)

    # Gather all images and masks
    samples = collect_samples(destination_directory)
    total_samples = len(samples)

    # Prepare erosion kernels once
    erosion_kernels = [None] + [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        for size in range(1, 30)
    ]

    for sample_id, sample in tqdm(samples.items(), total=total_samples, desc="Generating extras", unit="sample"):
        image_path = sample.get("image")
        mask_path = sample.get("mask")

        # Load image and mask
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        alpha = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            continue
        if alpha is None:
            logging.error(f"Failed to read mask: {mask_path}")
            continue

        # Normalize alpha to [0, 1]
        alpha = alpha.astype(np.float32) / 255.0

        ### Generate foreground and background ###
        image_float = image.astype(np.float32)
        alpha_3ch = np.expand_dims(alpha, axis=2)

        fg = image_float * (1.0 - alpha_3ch)
        bg = image_float * alpha_3ch

        # Save as PNG to preserve quality
        fg_path = fg_dest / f"{mask_path.stem}.png"
        bg_path = bg_dest / f"{mask_path.stem}.png"
        cv2.imwrite(str(fg_path), fg.astype(np.uint8))
        cv2.imwrite(str(bg_path), bg.astype(np.uint8))

        ### Generate trimap ###
        fg_width = np.random.randint(5, 15)
        bg_width = np.random.randint(5, 15)
        fg_mask = (alpha >= 0.9).astype(np.uint8)
        bg_mask = (alpha <= 0.1).astype(np.uint8)

        fg_eroded = cv2.erode(fg_mask, erosion_kernels[fg_width])
        bg_eroded = cv2.erode(bg_mask, erosion_kernels[bg_width])

        trimap = np.ones_like(alpha, dtype=np.uint8) * 128
        trimap[fg_eroded == 1] = 255
        trimap[bg_eroded == 1] = 0

        trimap_path = trimaps_dest / f"{mask_path.stem}.png"
        cv2.imwrite(str(trimap_path), trimap)


def main() -> None:
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Directories
    destination_directory = Path(configuration.destination_directory)

    # Process the data
    generate_extras(destination_directory)


if __name__ == '__main__':
    main()
