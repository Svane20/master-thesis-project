import bpy
from pathlib import Path
from typing import Tuple
import logging

from bpy_utils.bpy_data import list_data_blocks_in_blend_file, BlendFilePropertyKey
from constants.file_extensions import FileExtension


def delete_object_by_selection(obj: bpy.types.Object) -> None:
    """
    Deletes the given object after selection in the Blender scene.

    Args:
        obj (bpy.types.Object): The object to delete.

    Raises:
        RuntimeError: If the object fails to delete.
    """
    try:
        logging.info(f"Deleting object: '{obj.name}'")

        obj.select_set(True)
        bpy.ops.object.delete()

        logging.info(f"Successfully deleted object: '{obj.name}'")
    except Exception as e:
        logging.error(f"Failed to delete object: {e}")
        raise


def append_object(object_path: Path) -> bpy.types.Collection:
    """
    Appends the first collection from a .blend file to the scene.

    Args:
       object_path (Path): The path to the .blend file containing the object.

    Returns:
       bpy.types.Collection: The appended object collection.

    Raises:
       FileNotFoundError: If the .blend file or collection cannot be found.
       RuntimeError: If appending the object fails.
   """
    try:
        # Get the first collection's name
        collections_dict = list_data_blocks_in_blend_file(object_path, key=BlendFilePropertyKey.Collections)
        if not collections_dict:
            raise FileNotFoundError(f"No collections found in {object_path}")
        collection_name = next(iter(collections_dict.keys()))

        # Construct filepaths
        file_path, collection_path = _build_append_paths(object_path, collection_name)

        logging.info(f"Appending object '{collection_name}' from collection '{collection_path}'")

        # Append the object from the collection
        bpy.ops.wm.append(
            filepath=file_path,
            directory=collection_path,
            filename=collection_name
        )

        logging.info(f"Successfully appended collection: '{collection_name}'")

        return bpy.data.collections[collection_name]

    except Exception as e:
        logging.error(f"Failed to append object: {e}")
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
        logging.info("Rendering image...")

        # Render the image
        bpy.ops.render.render(write_still=write_still)

        logging.info("Image rendered successfully.")
    except Exception as e:
        logging.error(f"Failed to render image: {e}")
        raise


def save_as_blend_file(directory_path: str, iteration: int = 0, allow_overwrite: bool = True) -> None:
    """
    Saves the current Blender scene as a .blend file.

    Args:
        iteration (int): The current iteration number. Defaults to 0.
        directory_path (Path): The directory path to save the .blend file.
        allow_overwrite (bool): Whether to allow overwriting an existing file. Defaults to True.

    Raises:
        IOError: If the blend file fails to save.
    """
    filename = f"Blend_{iteration}.{FileExtension.BLEND.value}"
    logging.info(f"Saving blend file: '{filename}'")

    try:
        directory_path = Path(directory_path)
        output_path = _prepare_output_path(filename, directory_path, allow_overwrite)

        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        logging.info(f"Blend file saved successfully: '{filename}'")
    except Exception as e:
        logging.error(f"Failed to save blend file: {e}")
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


def _prepare_output_path(filename: str, directory_path: Path, allow_overwrite: bool) -> Path:
    """
    Prepares the output path for saving a .blend file.

    Args:
        filename (str): The name of the file to save.
        directory_path (Path): The directory path to save the .blend file.
        allow_overwrite (bool): Whether to allow overwriting an existing file.

    Returns:
        Path: The prepared output file path.

    Raises:
        FileNotFoundError: If the directory does not exist and cannot be created.
    """
    output_dir = directory_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if allow_overwrite and output_path.exists():
        output_path.unlink()

    return output_path
