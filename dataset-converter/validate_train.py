import logging
from pathlib import Path

from configuration.root import get_configurations
from custom_logging.custom_logger import setup_logging
from utils import collect_samples_extra

def validate_train_directory(path: Path) -> None:
    samples = collect_samples_extra(path)
    total_samples = len(samples)
    logging.info(f"Found {total_samples} samples in the train directory.")

    for sample_id, sample in samples.items():
        image = sample.get("image")
        mask = sample.get("mask")
        trimap = sample.get("trimap")
        fg = sample.get("fg")
        bg = sample.get("bg")

        # Check image existence and log the folder and filename if missing
        if image is None or not image.exists():
            image_name = image.name if image else "Unknown image name"
            folder_name = image.parent if image else "Unknown folder"
            logging.error(f"Missing image file: '{image_name}' in folder: '{folder_name}' for sample: '{sample_id}'")

        # Check mask existence and log the folder and filename if missing
        if mask is None or not mask.exists():
            mask_name = mask.name if mask else "Unknown mask name"
            folder_name = mask.parent if mask else "Unknown folder"
            logging.error(f"Missing mask file: '{mask_name}' in folder: '{folder_name}' for sample: '{sample_id}'")

        # Check trimap existence and log the folder and filename if missing
        if trimap is None or not trimap.exists():
            trimap_name = trimap.name if trimap else "Unknown trimap name"
            folder_name = trimap.parent if trimap else "Unknown folder"
            logging.error(f"Missing trimap file: '{trimap_name}' in folder: '{folder_name}' for sample: '{sample_id}'")

        # Check fg existence and log the folder and filename if missing
        if fg is None or not fg.exists():
            fg_name = fg.name if fg else "Unknown fg name"
            folder_name = fg.parent if fg else "Unknown folder"
            logging.error(f"Missing foreground file: '{fg_name}' in folder: '{folder_name}' for sample: '{sample_id}'")

        if bg is None or not bg.exists():
            bg_name = bg.name if bg else "Unknown bg name"
            folder_name = bg.parent if bg else "Unknown folder"
            logging.error(f"Missing background file: '{bg_name}' in folder: '{folder_name}' for sample: '{sample_id}'")

if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Train directory
    root_dir = Path(configuration.destination_directory)
    train_dir = root_dir / "train"

    # Validate the train directory
    validate_train_directory(train_dir)