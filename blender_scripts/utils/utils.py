from typing import Union
from pathlib import Path

from configuration.configuration import RenderConfiguration
from configuration.consts import Constants
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def get_temporary_file_path(file_name: Union[str | None], render_configuration: RenderConfiguration) -> str:
    """
    Get the path to a temporary file.

    Args:
        file_name: The name of the file.
        render_configuration: The render configuration.

    Returns:
        The path to the temporary file.
    """
    temp_dir: Path = Path(render_configuration.tmp_folder)

    path = temp_dir / (file_name or "render")
    path.parent.mkdir(parents=True, exist_ok=True)

    return path.as_posix()


def remove_temporary_files(
        directory: Path,
        image_name: str = None,
        extension: str = Constants.FileExtension.PNG
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
