from pathlib import Path
import logging
import shutil
from typing import Dict
from tqdm.auto import tqdm

from configuration.root import get_configurations
from custom_logging.custom_logger import setup_logging


def _collect_samples(source_directory: Path) -> Dict[str, Dict[str, Path]]:
    """
    Collect the samples from the source directory.

    Args:
        source_directory (Path): The source directory containing the data.

    Returns:
        dict: A dictionary containing the samples.
    """
    samples = {}

    for folder in source_directory.iterdir():
        if folder.is_dir():
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            for image_file in images_folder.glob("*.jpg"):
                sample_id = f"{folder.name}_{image_file.name}"
                sample = {"image": image_file}

                if image_file.name.startswith("Image_"):
                    expected_mask_name = f"{image_file.stem.replace("Image_", "SkyMask_", 1)}.png"
                    expected_mask = masks_folder / expected_mask_name
                    if expected_mask.exists():
                        sample["mask"] = expected_mask

                samples[sample_id] = sample

    # Validate that each image has a corresponding mask and vice versa
    for sample_id, sample in samples.items():
        if "mask" not in sample:
            raise ValueError(f"Mask not found for image: {sample['image']}")
        if "image" not in sample:
            raise ValueError(f"Image not found for mask: {sample['mask']}")

    return samples


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
    samples = _collect_samples(source_directory)
    total_samples = len(samples)

    # Process the data
    counter = 0
    for sample in tqdm(samples.values(), total=total_samples, desc="Processing samples"):
        img = sample.get("image")
        msk = sample.get("mask")

        if not img:
            logging.error("Image not found for a sample.")
            continue
        if not msk:
            logging.error(f"Mask not found for image: {img}")
            continue

        new_base = f"{counter:04d}"
        shutil.copy(img, images_dest / f"{new_base}{img.suffix}")
        shutil.copy(msk, masks_dest / f"{new_base}{msk.suffix}")

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
