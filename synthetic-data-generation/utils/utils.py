from datetime import datetime, timezone
from pathlib import Path
import logging

from configuration.configuration import Configuration
from configuration.outputs import OutputsConfiguration
from configuration.render import RenderConfiguration
from constants.file_extensions import FileExtension


class Constants:
    IMAGES_DIRNAME = "images"
    MASKS_DIRNAME = "masks"
    BLENDER_FILES_DIRNAME = "blender_files"


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
        logging.info(f"Created directory: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


def get_temporary_file_path(render_configuration: RenderConfiguration) -> str:
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


def get_playground_directory_with_tag(configuration: Configuration) -> Path:
    """
    Get the playground directory with a unique tag based on the current timestamp.

    Args:
        configuration (Configuration): The configuration object

    Returns:
        Path: The path to the playground directory.
    """
    current_time = datetime.now(timezone.utc).strftime("%H_%M_%d-%m-%Y")
    directory = Path(configuration.directories.playground_directory) / current_time
    create_directory(directory)

    # Create subdirectories for blender files, images and masks
    if configuration.constants.save_blend_files:
        create_directory(directory / Constants.BLENDER_FILES_DIRNAME)

    if configuration.constants.render_images:
        create_directory(directory / Constants.IMAGES_DIRNAME)
        create_directory(directory / Constants.MASKS_DIRNAME)

    return directory


def move_rendered_images_to_playground(
        configuration: OutputsConfiguration,
        directory: Path,
        iteration: int,
) -> None:
    """
    Move rendered images from the output directory to the destination directory.

    Args:
        configuration (OutputsConfiguration): The outputs configuration.
        directory (Path): The destination directory.
        iteration (int): The current iteration number.

    Raises:
        Exception: If the image moving fails.
    """
    file_extension = FileExtension.PNG.value
    rendered_images = Path(configuration.output_path).glob(f"*.{file_extension}")

    image_prefix = configuration.image_output_configuration.title
    id_mask_prefix = configuration.id_mask_output_configuration.title
    environment_prefix = configuration.environment_output_configuration.title

    for image in rendered_images:
        try:
            if image_prefix in image.name:
                filepath = _get_playground_file_path(directory, Constants.IMAGES_DIRNAME, image_prefix, iteration,
                                                     file_extension)
                image.rename(filepath)
                logging.info(f"Moved {image.name} to {filepath}")
            elif id_mask_prefix in image.name:
                filepath = _get_playground_file_path(
                    directory,
                    Constants.MASKS_DIRNAME,
                    id_mask_prefix,
                    iteration,
                    file_extension
                )
                image.rename(filepath)
                logging.info(f"Moved {image.name} to {filepath}")
            elif environment_prefix in image.name:
                filepath = _get_playground_file_path(
                    directory,
                    Constants.MASKS_DIRNAME,
                    environment_prefix,
                    iteration,
                    file_extension
                )
                image.rename(filepath)
                logging.info(f"Moved {image.name} to {filepath}")
        except Exception as e:
            logging.error(f"Failed to move {image.name} to {directory}: {e}")
            raise


def cleanup_files(configuration: Configuration, remove_output_dir: bool = True,
                  remove_temporary_dir: bool = True) -> None:
    """
    Clean up files from output, temporary, and Blender directories.

    Args:
        configuration (Configuration): The configuration object.
        remove_output_dir (bool, optional): Whether to remove files in the output directory. Defaults to True.
        remove_temporary_dir (bool, optional): Whether to remove files in the temporary directory. Defaults to True.
    """
    if remove_output_dir:
        output_directory = Path(configuration.render_configuration.outputs_configuration.output_path)
        logging.info(f"Cleaning up output directory: {output_directory}")
        remove_temporary_files(directory=output_directory)

    if remove_temporary_dir:
        temp_directory = Path(configuration.render_configuration.temp_folder)
        logging.info(f"Cleaning up temporary directory: {temp_directory}")
        remove_temporary_files(directory=temp_directory)


def remove_temporary_files(
        directory: Path,
        image_name: str = None,
        extension: FileExtension = FileExtension.PNG
) -> None:
    """
    Remove temporary files matching the specified extension and pattern in the directory.

    Args:
        directory (Path): The directory to search for files.
        image_name (str, optional): The name of the image. If None, delete all files with the specified extension.
        extension (FileExtension, optional): The file extension to filter by. Defaults to PNG.

    Raises:
        Exception: If the file deletion fails.
    """
    # Convert FileExtension to string
    extension = extension.value

    try:
        if image_name is None:
            logging.info(f"Deleting all temporary files with extension {extension} in {directory}")
            temp_files = directory.glob(f"*.{extension}")
        else:
            logging.info(f"Deleting temporary files with name {image_name}.{extension} in {directory}")
            temp_files = directory.glob(f"{image_name}.{extension}")

        for temp_file in temp_files:
            temp_file.unlink()
            logging.info(f"Deleted temporary file: {temp_file}")
    except Exception as e:
        logging.error(f"Failed to delete files in {directory}: {e}")
        raise


def _get_playground_file_path(
        directory: Path,
        subdirectory: str,
        output_name: str,
        iteration: int,
        file_extension: str
) -> str:
    """
    Get the path to a file in the playground directory.

    Args:
        directory (Path): The playground directory.
        subdirectory (str): The subdirectory name.
        output_name (str): The name of the output file.
        iteration (int): The current iteration number.
        file_extension (str): The file extension.

    Returns:
        str: The path to the file in the playground directory.
    """
    return (directory / f"{subdirectory}" / f"{output_name}_{iteration}.{file_extension}").as_posix()
