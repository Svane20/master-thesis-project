import bpy
from pathlib import Path
import logging

from configuration.configuration import Configuration
from custom_logging.custom_logger import setup_logging
from main import get_configuration


def check_alpha_matte_in_directory(configuration: Configuration) -> None:
    """
    Check if the alpha channel of the skyMask images in the source directory are valid continuous alpha mattes.
    For our purposes, a valid continuous matte should:
      - Have 4 channels (RGBA).
      - Have an alpha channel with values normalized between 0 and 1 that spans a significant range,
        e.g. min(alpha) < 0.1 and max(alpha) > 0.9.
      - Optionally, log the unique RGB values.
      
    Args:
        configuration (Configuration): The configuration for the validation pipeline.
    """
    source_directory = Path(configuration.constants.playground_directory)

    for mask_path in source_directory.rglob("*.png"):
        # Process only sky or grass mask images.
        if "SkyMask" not in mask_path.name:
            continue

        logging.info(f"Processing mask: {mask_path}")

        try:
            # Load the image via Blender's bpy.
            mask_image = bpy.data.images.load(str(mask_path))
        except Exception as e:
            logging.error(f"Error loading {mask_path}: {e}")
            continue

        # Set the image to non-color space to get raw pixel data.
        mask_image.colorspace_settings.name = 'Non-Color'

        # Check that the image has 4 channels (RGBA) and valid dimensions.
        if mask_image.channels != 4 or mask_image.size[0] == 0 or mask_image.size[1] == 0:
            logging.error(
                f"{mask_path} does not have 4 channels (found {mask_image.channels}) or has invalid dimensions. Skipping.")
            bpy.data.images.remove(mask_image)
            continue

        # Get the pixel data.
        # Blender stores pixels in a flat list: [R, G, B, A, R, G, B, A, ...]
        pixels = list(mask_image.pixels)
        channels = 4  # We expect RGBA

        # Collect alpha values.
        alpha_values = [pixels[i + 3] for i in range(0, len(pixels), channels)]
        min_alpha = min(alpha_values)
        max_alpha = max(alpha_values)
        logging.info(f"Alpha range in {mask_path}: min {min_alpha:.3f}, max {max_alpha:.3f}")

        # Determine if the matte is valid:
        # For example, we require that the matte covers a significant range: near 0 for geometry and near 1 for sky.
        if min_alpha < 0.1 and max_alpha > 0.9:
            logging.info(f"{mask_path} is a valid continuous alpha matte.")
        else:
            logging.error(f"{mask_path} does not span a sufficient range: min {min_alpha:.3f}, max {max_alpha:.3f}")

        # Remove the image from memory after processing.
        bpy.data.images.remove(mask_image)


def main():
    # Setup logging.
    setup_logging(__name__)

    # Load configuration.
    configuration = get_configuration()

    # Check the alpha mattes for sky masks in the source directory.
    check_alpha_matte_in_directory(configuration)


if __name__ == "__main__":
    main()
