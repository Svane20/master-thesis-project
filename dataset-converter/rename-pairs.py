from pathlib import Path
import logging
import os

from configuration.base import get_configurations
from custom_logging.custom_logger import setup_logging


def extract_image_key(stem: str):
    """
    Extract a key from an image filename that follows the pattern:
    <prefix>_Image_<number>
    Returns a tuple (prefix, number) if found; otherwise, None.
    """
    parts = stem.split('_')
    if "Image" in parts:
        idx = parts.index("Image")
        if idx + 1 < len(parts):
            number = parts[idx + 1]
            prefix = "_".join(parts[:idx])
            return (prefix, number)
    return None


def extract_mask_key(stem: str):
    """
    Extract a key from a mask filename that follows the pattern:
    <prefix>_SkyMask_<number>
    Returns a tuple (prefix, number) if found; otherwise, None.
    """
    parts = stem.split('_')
    if "SkyMask" in parts:
        idx = parts.index("SkyMask")
        if idx + 1 < len(parts):
            number = parts[idx + 1]
            prefix = "_".join(parts[:idx])
            return (prefix, number)
    return None


def main() -> None:
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()

    source_dir = Path(configuration.source_directory)
    images_dir = source_dir / "images"
    masks_dir = source_dir / "masks"

    # === Pre-renaming check ===
    pre_image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
    pre_mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file()])

    num_images = len(pre_image_files)
    num_masks = len(pre_mask_files)
    logging.info(f"Number of images: {num_images}")
    logging.info(f"Number of masks: {num_masks}")

    if num_images != num_masks:
        logging.warning("Mismatch in the number of images and masks!")
    else:
        logging.info("Pre-renaming check: Equal number of images and masks.")

    # Separate files into those following the pattern and numeric ones.
    image_pattern_dict = {}
    numeric_images = []
    for file in pre_image_files:
        stem = file.stem
        key = extract_image_key(stem)
        if key:
            image_pattern_dict[key] = file
        elif stem.isdigit():
            numeric_images.append(file)
        else:
            logging.warning(f"Image file {file.name} does not match expected patterns.")

    mask_pattern_dict = {}
    numeric_masks = []
    for file in pre_mask_files:
        stem = file.stem
        key = extract_mask_key(stem)
        if key:
            mask_pattern_dict[key] = file
        elif stem.isdigit():
            numeric_masks.append(file)
        else:
            logging.warning(f"Mask file {file.name} does not match expected patterns.")

    # Check that for each patterned image there is a corresponding mask.
    for key, image_file in image_pattern_dict.items():
        if key not in mask_pattern_dict:
            logging.warning(f"Missing mask for image with key {key} (from file {image_file.name}).")
    for key, mask_file in mask_pattern_dict.items():
        if key not in image_pattern_dict:
            logging.warning(f"Missing image for mask with key {key} (from file {mask_file.name}).")

    # For numeric files, check that the counts match.
    if len(numeric_images) != len(numeric_masks):
        logging.warning("Mismatch in the number of numeric image and mask files!")
    else:
        logging.info("Numeric file pairing (by count) looks good.")

    # === Renaming Process ===
    counter = 0
    for img, msk in zip(pre_image_files, pre_mask_files):
        new_base = f"{counter:04d}"  # New base name, e.g., "0000"
        new_img_name = new_base + img.suffix
        new_mask_name = new_base + msk.suffix

        new_img_path = images_dir / new_img_name
        new_mask_path = masks_dir / new_mask_name

        os.rename(img, new_img_path)
        os.rename(msk, new_mask_path)
        logging.info(f"Renamed '{img.name}' and '{msk.name}' to '{new_img_name}' and '{new_mask_name}'.")
        counter += 1

    # === Post-renaming check ===
    post_image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
    post_mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file()])

    # Now that we've renamed, the stems should match.
    post_image_bases = {f.stem for f in post_image_files}
    post_mask_bases = {f.stem for f in post_mask_files}

    if post_image_bases == post_mask_bases:
        logging.info("Post-renaming check passed: All images have corresponding masks with matching stems.")
    else:
        missing_masks = post_image_bases - post_mask_bases
        missing_images = post_mask_bases - post_image_bases
        if missing_masks:
            logging.warning(f"Missing masks for image(s) with base name(s): {', '.join(sorted(missing_masks))}")
        if missing_images:
            logging.warning(f"Missing images for mask(s) with base name(s): {', '.join(sorted(missing_images))}")


if __name__ == "__main__":
    main()
