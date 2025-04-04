from pathlib import Path
from typing import Dict


def collect_samples(directory: Path) -> Dict[str, Dict[str, Path]]:
    """
    Collect the samples from the specified directory.

    Args:
        directory (Path): The directory containing the data.

    Returns:
        Dict[str, Dict[str, Path]]: A dictionary containing the samples.
    """
    samples = {}

    for folder in directory.iterdir():
        if folder.is_dir():
            images_folder = folder / "images"
            masks_folder = folder / "masks"

            for image_file in sorted(images_folder.glob("*.jpg")):
                sample_id = f"{folder.name}_{image_file.name}"
                sample = {"image": image_file}

                if image_file.name.startswith("Image_"):
                    expected_mask_name = f"{image_file.stem.replace("Image_", "SkyMask_", 1)}.png"
                    expected_mask = masks_folder / expected_mask_name
                    if expected_mask.exists():
                        sample["mask"] = expected_mask
                else:
                    sample["mask"] = masks_folder / f"{image_file.stem}.png"

                samples[sample_id] = sample

    # Check for missing corresponding files
    for sample_id, sample in samples.items():
        if "mask" not in sample:
            raise ValueError(
                f"Missing mask for image '{sample['image'].name}' in folder '{sample['image'].parent}' for sample: '{sample_id}'"
            )
        if "image" not in sample:
            raise ValueError(
                f"Missing image for mask '{sample['mask'].name}' in folder '{sample['mask'].parent}' for sample: '{sample_id}'"
            )

    return samples

def collect_samples_extra(directory: Path) -> Dict[str, Dict[str, Path]]:
    """
    Collect the samples from the specified directory.

    Args:
        directory (Path): The directory containing the data.

    Returns:
        Dict[str, Dict[str, Path]]: A dictionary containing the samples.
    """
    samples = {}

    images_folder = directory / "images"
    masks_folder = directory / "masks"
    trimaps_folder = directory / "trimaps"
    fg_folder = directory / "fg"
    bg_folder = directory / "bg"

    for image_file in sorted(images_folder.glob("*.jpg")):
        sample_id = f"{directory.name}_{image_file.name}"

        sample = {"image": image_file}
        sample["mask"] = masks_folder / f"{image_file.stem}.png"
        sample["trimap"] = trimaps_folder / f"{image_file.stem}.png"
        sample["fg"] = fg_folder / f"{image_file.stem}.png"
        sample["bg"] = bg_folder / f"{image_file.stem}.png"

        samples[sample_id] = sample

    # Check for missing corresponding files
    for sample_id, sample in samples.items():
        if "mask" not in sample:
            raise ValueError(
                f"Missing mask for image '{sample['image'].name}' in folder '{sample['image'].parent}' for sample: '{sample_id}'"
            )
        if "image" not in sample:
            raise ValueError(
                f"Missing image for mask '{sample['mask'].name}' in folder '{sample['mask'].parent}' for sample: '{sample_id}'"
            )
        if "trimap" not in sample:
            raise ValueError(
                f"Missing trimap for image '{sample['image'].name}' in folder '{sample['image'].parent}' for sample: '{sample_id}'"
            )
        if "fg" not in sample:
            raise ValueError(
                f"Missing fg for image '{sample['image'].name}' in folder '{sample['image'].parent}' for sample: '{sample_id}'"
            )
        if "bg" not in sample:
            raise ValueError(
                f"Missing bg for image '{sample['image'].name}' in folder '{sample['image'].parent}' for sample: '{sample_id}'"
            )

    return samples
