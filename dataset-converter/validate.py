import logging
from pathlib import Path

from configuration.base import get_configurations
from custom_logging.custom_logger import setup_logging

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def validate_base_directory(source_dir: Path) -> None:
    """
    Validate that every image in the base directory has a corresponding mask and every mask has a corresponding image.
    It iterates over each folder in source_dir and checks the "images" and "masks" subdirectories.

    The naming convention assumes that:
      - Images are named like "Image_0.png"
      - Masks are named like "SkyMask_0.png"

    Args:
        source_dir (Path): Path to the top-level unprocessed data directory.
    """
    valid = True

    for folder in source_dir.iterdir():
        if folder.is_dir():
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            # Check that the expected subdirectories exist.
            if not images_folder.exists():
                logging.error(f"Missing images folder in {folder}")
                valid = False
            if not masks_folder.exists():
                logging.error(f"Missing masks folder in {folder}")
                valid = False
                continue

            # Collect files using allowed extensions.
            image_files = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            mask_files = [f for f in masks_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]

            # Build dictionaries mapping the file stem (e.g. "0001") to the file.
            image_dict = {f.stem: f for f in image_files}
            mask_dict = {f.stem: f for f in mask_files}

            # Verify every image has a corresponding mask.
            for stem, image_file in image_dict.items():
                if stem not in mask_dict:
                    logging.error(f"Missing mask for image {image_file.name} in folder {folder.name}")
                    valid = False

            # Verify every mask has a corresponding image.
            for stem, mask_file in mask_dict.items():
                if stem not in image_dict:
                    logging.error(f"Missing image for mask {mask_file.name} in folder {folder.name}")
                    valid = False

    if valid:
        logging.info("Base directory validation succeeded: Every image has a corresponding mask and vice versa.")
    else:
        logging.error("Base directory validation failed: Some images or masks are missing corresponding pairs.")


def validate_destination_directory(destination_dir: Path) -> None:
    """
    Validate that every image in the destination directory's train and test subfolders
    has a corresponding mask and every mask has a corresponding image.

    It assumes the following structure:
      destination_dir/
          train/
              images/
              masks/
          test/
              images/
              masks/

    The naming convention assumes that:
      - Images are named like "Image_0.png"
      - Masks are named like "SkyMask_0.png"

    Args:
        destination_dir (Path): Path to the top-level processed data directory.
    """
    valid = True

    for folder in destination_dir.iterdir():
        if folder.is_dir():
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            # Check that the expected subdirectories exist.
            if not images_folder.exists():
                logging.error(f"Missing images folder in {folder}")
                valid = False
            if not masks_folder.exists():
                logging.error(f"Missing masks folder in {folder}")
                valid = False
                continue

            # Collect files using allowed extensions.
            image_files = [f for f in images_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            mask_files = [f for f in masks_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]

            # Build dictionaries mapping the file stem (e.g. "0001") to the file.
            image_dict = {f.stem: f for f in image_files}
            mask_dict = {f.stem: f for f in mask_files}

            # Verify every image has a corresponding mask.
            for stem, image_file in image_dict.items():
                if stem not in mask_dict:
                    logging.error(f"Missing mask for image {image_file.name} in folder {folder.name}")
                    valid = False

            # Verify every mask has a corresponding image.
            for stem, mask_file in mask_dict.items():
                if stem not in image_dict:
                    logging.error(f"Missing image for mask {mask_file.name} in folder {folder.name}")
                    valid = False

    if valid:
        logging.info("Base directory validation succeeded: Every image has a corresponding mask and vice versa.")
    else:
        logging.error("Base directory validation failed: Some images or masks are missing corresponding pairs.")


if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Validate the base and destination directory
    validate_base_directory(Path(configuration.source_directory))
    validate_destination_directory(Path(configuration.destination_directory))
