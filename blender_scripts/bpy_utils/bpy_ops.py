import bpy

from pathlib import Path

from bpy_utils.bpy_data import list_data_blocks_in_blend_file, BlendFilePropertyKey
from constants.directories import BLENDER_FILES_DIRECTORY
from constants.file_extensions import FileExtension
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


def append_object(object_path: Path) -> bpy.types.Object:
    """
    Appends the first available object from a collection in a .blend file to the scene.

    Args:
        object_path: The path to the .blend file containing the object.

    Returns:
        The appended object.

    Raises:
        Exception: If the object fails to append.
    """
    try:
        # Get the first collection's name
        collections_dict = list_data_blocks_in_blend_file(object_path, key=BlendFilePropertyKey.Collections)
        collection_name = next(iter(collections_dict.keys()))

        # Construct filepaths
        file_path = str(object_path / "Collection" / f"{collection_name}")
        collection_path = str(object_path / "Collection" / "")  # Needs the trailing slash

        logger.info(f"Appending object '{collection_name}' from collection '{collection_path}'")

        # Append the object from the collection
        bpy.ops.wm.append(
            filepath=file_path,
            directory=collection_path,
            filename=collection_name
        )

        logger.info(f"Successfully appended object: '{collection_name}'")

        return bpy.data.collections[collection_name]

    except Exception as e:
        logger.error(f"Failed to append object: {e}")
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
        directory_path: Path = BLENDER_FILES_DIRECTORY,
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

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_name}.{FileExtension.BLEND.value}"

        # Remove the existing file if it exists
        if allow_overwrite and output_path.exists():
            output_path.unlink()

        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        logger.info(f"Saved: '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save blend file: {e}")
        raise
