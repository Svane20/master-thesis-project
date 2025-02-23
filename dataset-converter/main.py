from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm
import random

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


def flatten_and_split_dataset(source_dir: Path, dest_dir: Path, train_ratio: float = 0.8) -> None:
    """
    Flatten versioned synthetic data into train and validation directories,
    compressing PNG files with lossless compression to reduce file size.

    Creates train/val splits based on the given train_ratio (default: 80% train, 20% val).
    The directory structure in dest_dir will be:
        - train/images
        - train/masks
        - val/images
        - val/masks

    For each folder in the source, process images and masks, prepending the folder name to the filename
    for uniqueness.

    Args:
        source_dir (Path): Path to the top-level synthetic data folder.
        dest_dir (Path): Path to the destination folder where the flat structure will be created.
        train_ratio (float): Proportion of samples to be used for training.
    """
    logging.info("Starting dataset flattening and splitting...")

    # Create destination directories for train and validation splits
    train_images_dest = dest_dir / "train" / "images"
    train_masks_dest = dest_dir / "train" / "masks"
    val_images_dest = dest_dir / "val" / "images"
    val_masks_dest = dest_dir / "val" / "masks"

    for directory in [train_images_dest, train_masks_dest, val_images_dest, val_masks_dest]:
        directory.mkdir(parents=True, exist_ok=True)

    # Gather samples from source_dir
    # Each sample key will be a unique filename (foldername_filename.png)
    # and its value will be a dict with keys "image" and optionally "mask"
    samples = {}
    for folder in source_dir.iterdir():
        if folder.is_dir():
            logging.info(f"Processing folder: {folder.name}")
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            if images_folder.exists():
                for image_file in images_folder.glob('*.png'):
                    sample_id = f"{folder.name}_{image_file.name}"
                    samples.setdefault(sample_id, {})["image"] = image_file
            else:
                logging.info(f"No images folder found in {folder}")

            if masks_folder.exists():
                for mask_file in masks_folder.glob('*.png'):
                    sample_id = f"{folder.name}_{mask_file.name}"
                    samples.setdefault(sample_id, {})["mask"] = mask_file
            else:
                logging.info(f"No masks folder found in {folder}")

    total_samples = len(samples)
    logging.info(f"Total samples collected: {total_samples}")

    # Create train/val split
    sample_ids = list(samples.keys())
    random.shuffle(sample_ids)
    split_index = int(total_samples * train_ratio)
    train_samples = sample_ids[:split_index]
    val_samples = sample_ids[split_index:]

    logging.info(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Process training samples
    with tqdm(total=len(train_samples), desc="Processing training samples") as pbar:
        for sample_id in train_samples:
            sample = samples[sample_id]
            if "image" in sample:
                dest_image_path = train_images_dest / sample_id
                if not dest_image_path.exists():
                    compress_png(sample["image"], dest_image_path)
            if "mask" in sample:
                dest_mask_path = train_masks_dest / sample_id
                if not dest_mask_path.exists():
                    compress_png(sample["mask"], dest_mask_path)
            pbar.update(1)

    # Process validation samples
    with tqdm(total=len(val_samples), desc="Processing validation samples") as pbar:
        for sample_id in val_samples:
            sample = samples[sample_id]
            if "image" in sample:
                dest_image_path = val_images_dest / sample_id
                if not dest_image_path.exists():
                    compress_png(sample["image"], dest_image_path)
            if "mask" in sample:
                dest_mask_path = val_masks_dest / sample_id
                if not dest_mask_path.exists():
                    compress_png(sample["mask"], dest_mask_path)
            pbar.update(1)

    logging.info("Finished dataset flattening and splitting")


if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Directories
    source_directory = Path(configuration.source_directory)
    destination_directory = Path(configuration.destination_directory)

    # Flatten and split dataset (80% train, 20% val)
    flatten_and_split_dataset(source_directory, destination_directory)
