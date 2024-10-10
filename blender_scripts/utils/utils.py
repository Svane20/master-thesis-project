from datetime import datetime
from pathlib import Path

from configuration.configuration import RenderConfiguration
from constants.directories import PLAYGROUND_DIRECTORY, OUTPUT_DIRECTORY, TEMP_DIRECTORY, BLENDER_FILES_DIRECTORY
from constants.file_extensions import FileExtension

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def get_temporary_file_path(render_configuration: RenderConfiguration) -> str:
    """
    Get the path to a temporary file.

    Args:
        render_configuration: The render configuration.

    Returns:
        The path to the temporary file.
    """
    temp_dir: Path = Path(render_configuration.temp_folder)

    path = temp_dir / "temp"
    path.parent.mkdir(parents=True, exist_ok=True)

    return path.as_posix()


def get_playground_directory_with_tag(output_name: str = None) -> Path:
    """
    Get the playground directory with a unique tag.

    Args:
        output_name: The name of the output file. Defaults to None.

    Returns:
        The playground directory.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"{output_name}_{current_time}" if output_name is not None else current_time

    directory = PLAYGROUND_DIRECTORY / tag
    directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created playground directory: {directory}")

    return directory


def move_rendered_images_to_playground(
        playground_directory: Path,
        iteration: int,
        output_path: Path = OUTPUT_DIRECTORY,
) -> None:
    """
    Move rendered images to the playground directory.

    Args:
        playground_directory: The playground directory.
        iteration: The iteration number.
        output_path: The output directory
    """

    file_extension = FileExtension.PNG.value

    rendered_images = output_path.glob(f"*.{file_extension}")

    for image in rendered_images:
        try:
            if "Image" in image.name:
                filepath = (playground_directory / f"Image_{iteration}.{file_extension}").as_posix()
                image.rename(filepath)
                logger.info(f"Moved {image.name} to {filepath}")
            elif "IDMask" in image.name:
                filepath = (playground_directory / f"IDMask_{iteration}.{file_extension}").as_posix()
                image.rename(filepath)
                logger.info(f"Moved {image.name} to {filepath}")
        except Exception as e:
            logger.error(f"Could not move {image}: {e}")


def cleanup_directories(
        remove_output_dir: bool = True,
        remove_temporary_dir: bool = True,
        remove_blender_dir: bool = False
) -> None:
    """
    Cleanup the directories.

    Args:
        remove_output_dir: Whether to remove the output directory.
        remove_temporary_dir: Whether to remove the temporary directory.
        remove_blender_dir: Whether to remove the Blender directory.
    """
    if remove_output_dir:
        remove_temporary_files(directory=OUTPUT_DIRECTORY)

    if remove_temporary_dir:
        remove_temporary_files(directory=TEMP_DIRECTORY)

    if remove_blender_dir:
        remove_temporary_files(
            directory=BLENDER_FILES_DIRECTORY,
            extension=FileExtension.BLEND.value
        )


def remove_temporary_files(
        directory: Path,
        image_name: str = None,
        extension: str = FileExtension.PNG.value
) -> None:
    """
    Remove temporary files that match a specific pattern.

    Args:
        directory: The directory to search for temporary files.
        image_name: The name of the image.
        extension: The file extension.
    """
    if image_name is None:
        logger.info(f"\nDeleting all temporary files with extension {extension} in {directory}")

        for temp_file in directory.glob(f"*.{extension}"):
            try:
                temp_file.unlink()
                logger.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Could not delete {temp_file}: {e}")
                raise
    else:
        logger.info(f"\nDeleting temporary files with name {image_name}.{extension} in {directory}")

        for temp_file in directory.glob(f"{image_name}.{extension}"):
            try:
                temp_file.unlink()
                logger.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Could not delete {temp_file}: {e}")
                raise
