from pathlib import Path
from PIL import Image
import logging

from configuration.base import get_configurations
from custom_logging.custom_logger import setup_logging


def compress_png(source_file: Path, dest_file: Path) -> None:
    """
    Open a PNG image and save it with lossless compression.

    Uses Pillow's optimize flag and compress_level=9, which is lossless.

    Args:
        source_file (Path): Path to the source PNG file.
        dest_file (Path): Path to the destination PNG file.
    """
    with Image.open(source_file) as img:
        img.save(dest_file, format="PNG", optimize=True, compress_level=9)


def flatten_dataset(source_dir: Path, dest_dir: Path) -> None:
    """
    Flatten versioned synthetic data into a single images and masks directory,
    compressing PNG files lossless to reduce file size.

    For each datetime folder in the source, process images and masks, prepending
    the datetime folder name to the filename for uniqueness.

    Args:
        source_dir (Path): Path to the top-level synthetic data folder.
        dest_dir (Path): Path to the destination folder where the flat structure will be created.
    """
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Define destination subdirectories
    images_dest = dest_dir / "images"
    masks_dest = dest_dir / "masks"
    images_dest.mkdir(parents=True, exist_ok=True)
    masks_dest.mkdir(parents=True, exist_ok=True)

    # Iterate through each datetime folder in the source directory
    for folder in source_dir.iterdir():
        if folder.is_dir():
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            # Process images
            if images_folder.exists():
                for image_file in images_folder.glob('*.png'):
                    new_image_name = f"{folder.name}_{image_file.name}"
                    dest_image_path = images_dest / new_image_name
                    compress_png(image_file, dest_image_path)
                    logging.info(f"Processed and copied {image_file} to {dest_image_path}")

            # Process mask files
            if masks_folder.exists():
                for mask_file in masks_folder.glob("*.png"):
                    new_mask_name = f"{folder.name}_{mask_file.name}"
                    dest_mask_path = masks_dest / new_mask_name
                    compress_png(mask_file, dest_mask_path)
                    logging.info(f"Processed and copied {mask_file} to {dest_mask_path}")


if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Directories
    source_directory = Path(configuration.source_directory)
    destination_directory = Path(configuration.destination_directory)

    # Flatten dataset
    flatten_dataset(source_directory, destination_directory)
