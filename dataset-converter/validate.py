import logging
from pathlib import Path

from configuration.root import get_configurations
from custom_logging.custom_logger import setup_logging
from utils import collect_samples


def validate_directory(directory: Path) -> None:
    """
    Validate the directory by checking if all samples have corresponding image and mask files.

    Args:
        directory (Path): Path to the data directory.
    """
    valid = True
    is_source_dir = str(directory).__contains__("raw")

    samples = collect_samples(directory)
    total_samples = len(samples)
    logging.info(f"Found {total_samples} samples in the {'source' if is_source_dir else 'destination'} directory.")

    for sample_id, sample in samples.items():
        image = sample.get("image")
        mask = sample.get("mask")

        # Check image existence and log the folder and filename if missing
        if image is None or not image.exists():
            image_name = image.name if image else "Unknown image name"
            folder_name = image.parent if image else "Unknown folder"
            logging.error(f"Missing image file: '{image_name}' in folder: '{folder_name}' for sample: '{sample_id}'")
            valid = False

        # Check mask existence and log the folder and filename if missing
        if mask is None or not mask.exists():
            mask_name = mask.name if mask else "Unknown mask name"
            folder_name = mask.parent if mask else "Unknown folder"
            logging.error(f"Missing mask file: '{mask_name}' in folder: '{folder_name}' for sample: '{sample_id}'")
            valid = False

    if not valid:
        raise ValueError("Validation failed: some samples are missing required files. See error logs for details.")
    else:
        logging.info("Validation succeeded: Every image has a corresponding mask and vice versa.")


if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Validate the base and destination directory
    validate_directory(Path(configuration.source_directory))
    validate_directory(Path(configuration.destination_directory))
