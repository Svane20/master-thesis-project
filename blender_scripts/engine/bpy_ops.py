import bpy

from pathlib import Path

from configuration.consts import Constants
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def delete_object_by_selection(object: bpy.types.Object) -> None:
    """
    Deletes the given object after selection.

    Args:
        object: The object to delete.

    Raises:
        Exception: If the object fails to delete.
    """
    try:
        logger.info(f"Deleting object: '{object.name}'")

        object.select_set(True)
        bpy.ops.object.delete()
    except Exception as e:
        logger.error(f"Failed to delete object: {e}")
        raise


def render_image(write_still: bool = True) -> None:
    """
    Renders the current scene as an image.

    Args:
        write_still: Whether to write the image to disk.

    Raises:
        Exception: If the image fails to render.
    """
    try:
        logger.info("Rendering image...")

        # Render the image
        bpy.ops.render.render(write_still=write_still)

        logger.info("Rendered image.")
    except Exception as e:
        logger.error(f"Failed to render image: {e}")
        raise


def save_as_blend_file(
        image_name: str,
        directory_path: Path = Constants.Directory.BLENDER_FILES_DIR,
        allow_overwrite: bool = True
) -> None:
    """
    Saves the current Blender scene as a .blend file.

    Args:
        image_name: The name of the image.
        directory_path: The directory path.
        allow_overwrite: Whether to allow overwriting the file.

    Raises:
        Exception: If the blend file fails to save.
    """
    try:
        # Ensure the directory exists
        output_dir: Path = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_name}.{Constants.FileExtension.BLEND}"

        # Remove the existing file if it exists
        if allow_overwrite and output_path.exists():
            output_path.unlink()

        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        logger.info(f"Saved: '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save blend file: {e}")
        raise
