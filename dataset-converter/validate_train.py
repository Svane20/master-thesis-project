import logging
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import json

from configuration.root import get_configurations
from custom_logging.custom_logger import setup_logging
from utils import collect_samples_extra


def verify_image_file(file_path: Path) -> bool:
    """
    Try to open and verify an image file.
    Returns True if the image is valid, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that the image is not corrupted
        return True
    except Exception as e:
        logging.error(f"Invalid image file: {file_path} - {e}")
        return False


def validate_train_directory(path: Path) -> None:
    samples = collect_samples_extra(path)
    total_samples = len(samples)
    logging.info(f"Found {total_samples} samples in the train directory.")

    invalid_files = []  # List of dictionaries for invalid files
    output_json = Path("invalid_files.json")

    pbar = tqdm(samples.items(), total=total_samples, desc="Validating samples")
    for sample_id, sample in pbar:
        pbar.set_postfix({"current_id": sample_id})
        for key in ["image", "mask", "trimap", "fg", "bg"]:
            file_path = sample.get(key)
            if file_path is None or not file_path.exists():
                file_name = file_path.name if file_path else "Unknown"
                folder_name = file_path.parent if file_path else "Unknown folder"
                logging.error(f"Missing {key} file: '{file_name}' in folder: '{folder_name}' for sample: '{sample_id}'")
                invalid_files.append({
                    "sample_id": sample_id,
                    "key": key,
                    "reason": "Missing file",
                    "file_name": file_name,
                    "file_path": str(file_path) if file_path else "None"
                })
                with output_json.open("w") as f:
                    json.dump(invalid_files, f, indent=4)
            else:
                # Verify that the file can be opened and is not corrupted
                if not verify_image_file(file_path):
                    invalid_files.append({
                        "sample_id": sample_id,
                        "key": key,
                        "reason": "Invalid file",
                        "file_name": file_path.name,
                        "file_path": str(file_path)
                    })
                    with output_json.open("w") as f:
                        json.dump(invalid_files, f, indent=4)

    if invalid_files:
        logging.error("Invalid files found. See invalid_files.json for details.")
    else:
        logging.info("All files are valid.")


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