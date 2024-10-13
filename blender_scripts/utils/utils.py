from datetime import datetime
from pathlib import Path

from configuration.outputs import ImageOutputConfiguration, IDMaskOutputConfiguration, EnvironmentOutputConfiguration
from configuration.render import RenderConfiguration
from constants.directories import PLAYGROUND_DIRECTORY, OUTPUT_DIRECTORY, TEMP_DIRECTORY, BLENDER_FILES_DIRECTORY
from constants.file_extensions import FileExtension

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def create_directory(path: Path) -> None:
    """
    Helper function to create a directory if it does not exist.

    Args:
        path (Path): The directory path to create.

    Raises:
        Exception: If the directory creation fails.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def get_temporary_file_path(render_configuration) -> str:
    """
    Get the path to a temporary file based on the render configuration.

    Args:
        render_configuration (RenderConfiguration): The render configuration.

    Returns:
        str: The path to the temporary file.
    """
    temp_dir: Path = Path(render_configuration.temp_folder)
    path = temp_dir / "temp"

    create_directory(path.parent)

    return path.as_posix()


def get_playground_directory_with_tag(output_name: str = None) -> Path:
    """
    Get the playground directory with a unique tag based on the current timestamp.

    Args:
        output_name (str, optional): The name of the output file. Defaults to None.

    Returns:
        Path: The path to the playground directory.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"{output_name}_{current_time}" if output_name else current_time

    directory = PLAYGROUND_DIRECTORY / tag
    create_directory(directory)

    return directory


def move_rendered_images_to_playground(
        directory: Path,
        iteration: int,
        output_path: Path = OUTPUT_DIRECTORY,
) -> None:
    """
    Move rendered images from the output directory to the destination directory.

    Args:
        directory (Path): The destination directory.
        iteration (int): The current iteration number.
        output_path (Path, optional): The directory containing rendered images. Defaults to OUTPUT_DIRECTORY.

    Raises:
        Exception: If the image moving fails.
    """
    file_extension = FileExtension.PNG.value
    rendered_images = output_path.glob(f"*.{file_extension}")

    image_prefix = ImageOutputConfiguration().title
    id_mask_prefix = IDMaskOutputConfiguration().title
    environment_prefix = EnvironmentOutputConfiguration().title

    for image in rendered_images:
        try:
            if image_prefix in image.name:
                filepath = (directory / f"{image_prefix}_{iteration}.{file_extension}").as_posix()
                image.rename(filepath)
                logger.info(f"Moved {image.name} to {filepath}")
            elif id_mask_prefix in image.name:
                filepath = (directory / f"{id_mask_prefix}_{iteration}.{file_extension}").as_posix()
                image.rename(filepath)
                logger.info(f"Moved {image.name} to {filepath}")
            elif environment_prefix in image.name:
                filepath = (directory / f"{environment_prefix}_{iteration}.{file_extension}").as_posix()
                image.rename(filepath)
                logger.info(f"Moved {image.name} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to move {image.name} to {directory}: {e}")
            raise


def cleanup_files(
        remove_output_dir: bool = True,
        remove_temporary_dir: bool = True,
        remove_blender_dir: bool = False
) -> None:
    """
    Clean up files from output, temporary, and Blender directories.

    Args:
        remove_output_dir (bool, optional): Whether to remove files in the output directory. Defaults to True.
        remove_temporary_dir (bool, optional): Whether to remove files in the temporary directory. Defaults to True.
        remove_blender_dir (bool, optional): Whether to remove Blender files. Defaults to False.
    """
    if remove_output_dir:
        logger.info(f"Cleaning up output directory: {OUTPUT_DIRECTORY}")
        remove_temporary_files(directory=OUTPUT_DIRECTORY)

    if remove_temporary_dir:
        logger.info(f"Cleaning up temporary directory: {TEMP_DIRECTORY}")
        remove_temporary_files(directory=TEMP_DIRECTORY)

    if remove_blender_dir:
        logger.info(f"Cleaning up Blender files directory: {BLENDER_FILES_DIRECTORY}")
        remove_temporary_files(directory=BLENDER_FILES_DIRECTORY, extension=FileExtension.BLEND)


def remove_temporary_files(directory: Path, image_name: str = None, extension: FileExtension = FileExtension.PNG) -> None:
    """
    Remove temporary files matching the specified extension and pattern in the directory.

    Args:
        directory (Path): The directory to search for files.
        image_name (str, optional): The name of the image. If None, delete all files with the specified extension.
        extension (str, optional): The file extension to filter by. Defaults to PNG.

    Raises:
        Exception: If the file deletion fails.
    """
    # Convert FileExtension to string
    extension = extension.value

    try:
        if image_name is None:
            logger.info(f"Deleting all temporary files with extension {extension} in {directory}")
            temp_files = directory.glob(f"*.{extension}")
        else:
            logger.info(f"Deleting temporary files with name {image_name}.{extension} in {directory}")
            temp_files = directory.glob(f"{image_name}.{extension}")

        for temp_file in temp_files:
            temp_file.unlink()
            logger.info(f"Deleted temporary file: {temp_file}")
    except Exception as e:
        logger.error(f"Failed to delete files in {directory}: {e}")
        raise
