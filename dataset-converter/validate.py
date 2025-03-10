import logging
from pathlib import Path

from configuration.base import get_configurations
from custom_logging.custom_logger import setup_logging


def validate_base_directory(source_dir: Path) -> bool:
    """
    Validate that every image in the base directory has a corresponding mask and every mask has a corresponding image.
    It iterates over each folder in source_dir and checks the "images" and "masks" subdirectories.

    The naming convention assumes that:
      - Images are named like "Image_0.png"
      - Masks are named like "SkyMask_0.png"

    Args:
        source_dir (Path): Path to the top-level unprocessed data directory.

    Returns:
        bool: True if every image has a corresponding mask and every mask has an image, False otherwise.
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

            # Verify every image has a corresponding mask.
            for image_file in images_folder.glob("*.png"):
                expected_mask_name = image_file.name.replace("Image_", "SkyMask_", 1)
                expected_mask = masks_folder / expected_mask_name

                if not expected_mask.exists():
                    logging.error(f"Missing mask for image {image_file.name} in folder {folder.name}")
                    valid = False

            # Verify every mask has a corresponding image.
            for mask_file in masks_folder.glob("*.png"):
                expected_image_name = mask_file.name.replace("SkyMask_", "Image_", 1)
                expected_image = images_folder / expected_image_name

                if not expected_image.exists():
                    logging.error(f"Missing image for mask {mask_file.name} in folder {folder.name}")
                    valid = False

    if valid:
        logging.info("Base directory validation succeeded: Every image has a mask and every mask has an image.")
    else:
        logging.error("Base directory validation failed: Some images or masks are missing corresponding pairs.")

    return valid


def validate_destination_directory(destination_dir: Path) -> bool:
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

    Returns:
        bool: True if every image has a corresponding mask and every mask has an image, False otherwise.
    """
    valid = True

    for subset in ["train", "test"]:
        subset_folder = destination_dir / subset
        if not subset_folder.exists():
            logging.error(f"Missing {subset} folder in {destination_dir}")
            valid = False
            continue

        images_folder = subset_folder / "images"
        masks_folder = subset_folder / "masks"

        if not images_folder.exists():
            logging.error(f"Missing images folder in {subset_folder}")
            valid = False

        if not masks_folder.exists():
            logging.error(f"Missing masks folder in {subset_folder}")
            valid = False
            continue

        # Verify every image has a corresponding mask.
        for image_file in images_folder.glob("*.png"):
            expected_mask_name = image_file.name.replace("Image_", "SkyMask_", 1)
            expected_mask = masks_folder / expected_mask_name
            if not expected_mask.exists():
                logging.error(f"Missing mask for image {image_file.name} in folder {subset_folder.name}")
                valid = False

        # Verify every mask has a corresponding image.
        for mask_file in masks_folder.glob("*.png"):
            expected_image_name = mask_file.name.replace("SkyMask_", "Image_", 1)
            expected_image = images_folder / expected_image_name
            if not expected_image.exists():
                logging.error(f"Missing image for mask {mask_file.name} in folder {subset_folder.name}")
                valid = False

    if valid:
        logging.info(
            "Destination directory validation succeeded: Every image has a mask and every mask has an image in both test and train subfolders.")
    else:
        logging.error(
            "Destination directory validation failed: Some images or masks are missing corresponding pairs in the test and train subfolders.")

    return valid


if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Validate the base directory
    source_directory = Path(configuration.source_directory)
    if validate_base_directory(source_directory):
        logging.info("Base directory is complete and valid.")
    else:
        logging.error("Base directory is incomplete or invalid.")

    # Validate the destination directory
    destination_directory = Path(configuration.destination_directory)
    if validate_destination_directory(destination_directory):
        logging.info("Destination directory is complete and valid.")
    else:
        logging.error("Destination directory is incomplete or invalid.")
