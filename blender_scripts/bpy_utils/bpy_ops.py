from typing import Tuple

import bpy

from pathlib import Path

from bpy_utils.bpy_data import list_data_blocks_in_blend_file, BlendFilePropertyKey
from constants.directories import BLENDER_FILES_DIRECTORY
from constants.file_extensions import FileExtension
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def delete_object_by_selection(object: bpy.types.Object) -> None:
    """
    Deletes the given object after selection in the Blender scene.

    Args:
        object (bpy.types.Object): The object to delete.

    Raises:
        Exception: If the object fails to delete.
    """
    try:
        logger.info(f"Deleting object: '{object.name}'")

        object.select_set(True)
        bpy.ops.object.delete()

        logger.info(f"Successfully deleted object: '{object.name}'")
    except Exception as e:
        logger.error(f"Failed to delete object: {e}")
        raise


def append_object(object_path: Path) -> bpy.types.Collection:
    """
    Appends the first collection from a .blend file to the scene.

    Args:
        object_path (Path): The path to the .blend file containing the object.

    Returns:
        bpy.types.Collection: The appended object collection.

    Raises:
        Exception: If the object fails to append.
    """
    try:
        # Get the first collection's name
        collections_dict = list_data_blocks_in_blend_file(object_path, key=BlendFilePropertyKey.Collections)
        collection_name = next(iter(collections_dict.keys()))

        # Construct filepaths
        file_path, collection_path = _build_append_paths(object_path, collection_name)

        logger.info(f"Appending object '{collection_name}' from collection '{collection_path}'")

        # Append the object from the collection
        bpy.ops.wm.append(
            filepath=file_path,
            directory=collection_path,
            filename=collection_name
        )

        logger.info(f"Successfully appended collection: '{collection_name}'")

        return bpy.data.collections[collection_name]

    except Exception as e:
        logger.error(f"Failed to append object: {e}")
        raise


def render_image(write_still: bool = True) -> None:
    """
    Renders the current Blender scene as an image.

    Args:
        write_still (bool): Whether to write the rendered image to disk. Defaults to True.

    Raises:
        Exception: If the image fails to render.
    """
    try:
        logger.info("Rendering image...")

        # Render the image
        bpy.ops.render.render(write_still=write_still)

        logger.info("Image rendered successfully.")
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
        image_name (str): The name of the file to save.
        directory_path (Path): The directory path to save the .blend file.
        allow_overwrite (bool): Whether to allow overwriting an existing file. Defaults to True.

    Raises:
        Exception: If the blend file fails to save.
    """
    try:
        output_path = _prepare_output_path(image_name, directory_path, allow_overwrite)

        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        logger.info(f"Blend file saved successfully: '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save blend file: {e}")
        raise


def _build_append_paths(object_path: Path, collection_name: str) -> Tuple[str, str]:
    """
    Constructs the file path and directory path for appending a collection from a .blend file.

    Args:
        object_path (Path): The path to the .blend file.
        collection_name (str): The name of the collection to append.

    Returns:
        Tuple[str, str]: The file path and collection path as strings.
    """
    file_path = str(object_path / "Collection" / collection_name)
    collection_path = str(object_path / "Collection" / "")  # Needs the trailing slash

    return file_path, collection_path


def _prepare_output_path(image_name: str, directory_path: Path, allow_overwrite: bool) -> Path:
    """
    Prepares the output path for saving a .blend file.

    Args:
        image_name (str): The name of the file to save.
        directory_path (Path): The directory path to save the .blend file.
        allow_overwrite (bool): Whether to allow overwriting an existing file.

    Returns:
        Path: The prepared output file path.
    """
    output_dir = directory_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_name}.{FileExtension.BLEND.value}"

    if allow_overwrite and output_path.exists():
        output_path.unlink()

    return output_path
