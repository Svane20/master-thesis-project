from pathlib import Path
import random
from typing import Tuple
import cv2
import numpy as np
import logging

from ...configuration.configuration import Config


def get_random_image(configuration: Config, output_dir: Path, save_image: bool = True, seed: int = None) -> Tuple[np.ndarray, Path]:
    """
    Get a random image from the test set

    Args:
        configuration (Config): Configuration object
        output_dir (Path): Output directory
        save_image (bool): Whether to save the chosen image
        seed (int): Seed for random choice

    Returns:
        np.ndarray: Image
    """
    if seed is not None:
        random.seed(seed)

    dataset_path = Path(configuration.dataset.root) / configuration.dataset.name
    test_dir = dataset_path / "test"
    images_dir = test_dir / "images"
    image_files = [f for f in images_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]]
    chosen_image_path = random.choice(image_files)
    logging.info(f"Chosen image: {chosen_image_path}")
    image = cv2.imread(str(chosen_image_path))

    if save_image:
        filename = f"input{chosen_image_path.suffix}"
        cv2.imwrite(str(output_dir / filename), image)
        logging.info(f"Saved chosen image to {output_dir / filename}")

    return image, chosen_image_path
