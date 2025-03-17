from pathlib import Path
import logging
import shutil
from tqdm.auto import tqdm

from configuration.root import get_configurations
from custom_logging.custom_logger import setup_logging
from utils import collect_samples


def process_data(source_directory: Path, destination_directory: Path) -> None:
    """
    Process the data from the source directory and save the results to the destination directory.

    Args:
        source_directory (Path): The source directory containing the data.
        destination_directory (Path): The destination directory to save the processed data.
    """
    # Directories
    train_dest = destination_directory / "train"
    images_dest = train_dest / "images"
    masks_dest = train_dest / "masks"

    # Create directories
    images_dest.mkdir(parents=True, exist_ok=True)
    masks_dest.mkdir(parents=True, exist_ok=True)

    # Gather all images and masks
    samples = collect_samples(source_directory)
    total_samples = len(samples)

    # Process the data
    counter = 0
    for sample in tqdm(samples.values(), total=total_samples, desc="Processing samples"):
        image = sample.get("image")
        mask = sample.get("mask")

        new_base = f"{counter:04d}"
        shutil.copy(image, images_dest / f"{new_base}{image.suffix}")
        shutil.copy(mask, masks_dest / f"{new_base}{mask.suffix}")

        counter += 1

    # Validate that all samples were processed
    if counter != total_samples:
        logging.error("Not all samples were processed.")


def main() -> None:
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Directories
    source_directory = Path(configuration.source_directory)
    destination_directory = Path(configuration.destination_directory)

    # Process the data
    process_data(source_directory, destination_directory)


if __name__ == '__main__':
    main()
