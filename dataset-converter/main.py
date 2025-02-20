from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm

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
    logging.info("Starting dataset flattening...")

    # Create destination directory and subdirectories
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dest = dest_dir / "images"
    masks_dest = dest_dir / "masks"
    images_dest.mkdir(parents=True, exist_ok=True)
    masks_dest.mkdir(parents=True, exist_ok=True)

    # Pre-calculate total number of files (images + masks)
    total_files = 0
    for folder in source_dir.iterdir():
        if folder.is_dir():
            images_folder = folder / "images"
            masks_folder = folder / "masks"
            if images_folder.exists():
                total_files += len(list(images_folder.glob('*.png')))
            if masks_folder.exists():
                total_files += len(list(masks_folder.glob('*.png')))

    logging.info(f"Total files to process: {total_files}")

    # Process all files
    with tqdm(total=total_files, desc="Processing all files") as pbar:
        for folder in source_dir.iterdir():
            if folder.is_dir():
                logging.info(f"Processing folder: {folder.name}")
                images_folder = folder / "images"
                masks_folder = folder / "masks"

                if images_folder.exists():
                    image_files = list(images_folder.glob('*.png'))

                    for image_file in image_files:
                        new_image_name = f"{folder.name}_{image_file.name}"
                        dest_image_path = images_dest / new_image_name

                        if not dest_image_path.exists():
                            compress_png(image_file, dest_image_path)

                        pbar.update(1)
                else:
                    logging.info(f"No images folder found in {folder}")

                if masks_folder.exists():
                    mask_files = list(masks_folder.glob("*.png"))

                    for mask_file in mask_files:
                        new_mask_name = f"{folder.name}_{mask_file.name}"
                        dest_mask_path = masks_dest / new_mask_name

                        if not dest_mask_path.exists():
                            compress_png(mask_file, dest_mask_path)

                        pbar.update(1)
                else:
                    logging.info(f"No masks folder found in {folder}")

    logging.info("Finished dataset flattening")


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
