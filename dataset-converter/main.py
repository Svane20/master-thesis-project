from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def process_all_tasks(tasks, desc, max_workers):
    """
    Process a list of (source, destination) tasks concurrently.

    Args:
        tasks (list): List of tuples (source_path, destination_path).
        desc (str): Description to show in the progress bar.
        max_workers (int): Maximum number of parallel workers.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compress_png, source, dest) for source, dest in tasks]
        with tqdm(total=len(futures), desc=desc) as pbar:
            for future in as_completed(futures):
                future.result()  # Propagate exceptions if any
                pbar.update(1)


def flatten_and_split_dataset(source_dir: Path, dest_dir: Path, train_ratio: float = 0.8, max_workers: int = 2) -> None:
    """
    Flatten versioned synthetic data into train and validation directories,
    moving PNG files.

    Creates train/val splits based on the given train_ratio (default: 80% train, 20% val).
    The directory structure in dest_dir will be:
        - train/images
        - train/masks
        - val/images
        - val/masks

    For each folder in the source, the script processes images from the "images" subfolder.
    For every image (named like "Image_0.png"), it looks for a corresponding mask in the "masks" subfolder.
    The corresponding mask is assumed to have the same number in its name but with the prefix "SkyMask" (e.g., "SkyMask_0.png").

    Args:
        source_dir (Path): Path to the top-level synthetic data folder.
        dest_dir (Path): Path to the destination folder where the flat structure will be created.
        train_ratio (float): Proportion of samples to be used for training.
        max_workers (int): Maximum number of parallel workers.
    """
    logging.info("Starting dataset flattening and splitting...")

    # Create destination directories for train and validation splits
    train_images_dest = dest_dir / "train" / "images"
    train_masks_dest = dest_dir / "train" / "masks"
    val_images_dest = dest_dir / "val" / "images"
    val_masks_dest = dest_dir / "val" / "masks"

    for directory in [train_images_dest, train_masks_dest, val_images_dest, val_masks_dest]:
        directory.mkdir(parents=True, exist_ok=True)

    # Gather samples from source_dir.
    samples = {}
    for folder in source_dir.iterdir():
        if folder.is_dir():
            logging.info(f"Processing folder: {folder.name}")
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            if images_folder.exists():
                for image_file in images_folder.glob('*.png'):
                    sample_id = f"{folder.name}_{image_file.name}"
                    sample = {"image": image_file}

                    # Compute expected mask filename (e.g., Image_0.png -> SkyMask_0.png)
                    if masks_folder.exists() and image_file.name.startswith("Image_"):
                        expected_mask_name = image_file.name.replace("Image_", "SkyMask_", 1)
                        expected_mask = masks_folder / expected_mask_name
                        if expected_mask.exists():
                            sample["mask"] = expected_mask

                    samples[sample_id] = sample
            else:
                logging.info(f"No images folder found in {folder}")

    total_samples = len(samples)
    logging.info(f"Total samples collected: {total_samples}")

    # Create train/val split
    sample_ids = list(samples.keys())
    random.shuffle(sample_ids)
    split_index = int(total_samples * train_ratio)
    train_samples = sample_ids[:split_index]
    val_samples = sample_ids[split_index:]

    logging.info(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Build combined task list for training and validation splits
    all_tasks = []

    # Training tasks
    for sample_id in train_samples:
        sample = samples[sample_id]
        if "image" in sample:
            dest_image_path = train_images_dest / sample_id
            if not dest_image_path.exists():
                all_tasks.append((sample["image"], dest_image_path))
        if "mask" in sample:
            dest_mask_path = train_masks_dest / sample_id.replace("Image_", "SkyMask_", 1)
            if not dest_mask_path.exists():
                all_tasks.append((sample["mask"], dest_mask_path))

    # Validation tasks
    for sample_id in val_samples:
        sample = samples[sample_id]
        if "image" in sample:
            dest_image_path = val_images_dest / sample_id
            if not dest_image_path.exists():
                all_tasks.append((sample["image"], dest_image_path))
        if "mask" in sample:
            dest_mask_path = val_masks_dest / sample_id.replace("Image_", "SkyMask_", 1)
            if not dest_mask_path.exists():
                all_tasks.append((sample["mask"], dest_mask_path))

    logging.info(f"Total tasks to process: {len(all_tasks)}")

    # Process all tasks concurrently using a single executor
    process_all_tasks(all_tasks, "Processing all samples", max_workers)
    logging.info("Finished dataset flattening and splitting")


def validate_dataset_split(split_dir: Path) -> bool:
    """
    Validate that every image in the split (e.g., val or test) has an associated mask.
    The association is determined by the naming convention:
        For an image file named "folder_Image_0.png" in split_dir/images,
        the corresponding mask is expected to be named "folder_SkyMask_0.png" in split_dir/masks.

    Args:
        split_dir (Path): The split directory to validate (e.g., destination_directory / "val").

    Returns:
        bool: True if all images have their corresponding masks, False otherwise.
    """
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    valid = True

    for image_path in images_dir.glob("*.png"):
        expected_mask_name = image_path.name.replace("Image_", "SkyMask_", 1)
        expected_mask_path = masks_dir / expected_mask_name
        if not expected_mask_path.exists():
            logging.error(f"Missing mask for image {image_path.name} expected at {expected_mask_path}")
            valid = False

    if valid:
        logging.info(f"All images in {split_dir} have associated masks.")
    else:
        logging.error(f"Validation failed for {split_dir}.")

    return valid


if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    # Define source and destination directories
    source_directory = Path(configuration.source_directory)
    destination_directory = Path(configuration.destination_directory)

    # Flatten and split dataset (e.g., 80% train, 20% val)
    flatten_and_split_dataset(
        source_directory,
        destination_directory,
        train_ratio=configuration.train_ratio,
        max_workers=min(configuration.num_workers, 2)
    )

    # Validate the train and validation splits
    for split in ["train", "val"]:
        split_dir = destination_directory / split
        if validate_dataset_split(split_dir):
            logging.info(f"{split.capitalize()} split is complete and valid.")
        else:
            logging.error(f"There were issues with the {split} split.")
