from typing import Union
from pathlib import Path

from configs.configuration import RenderConfiguration
from consts import Constants


def get_temporary_file_path(file_name: Union[str | None], render_configuration: RenderConfiguration) -> str:
    temp_dir: Path = Path(render_configuration.tmp_folder)

    path = temp_dir / (file_name or "render")
    path.parent.mkdir(parents=True, exist_ok=True)

    return path.as_posix()


def remove_temporary_files(
        directory: Path = Constants.Directory.TEMP_DIR,
        image_name: str = None,
        extension: str = "png"
) -> None:
    """Remove temporary files that match a specific pattern."""
    if image_name is None:
        print(f"\nDeleting all temporary files with extension {extension} in {directory}")

        for temp_file in directory.glob(f"*.{extension}"):
            try:
                temp_file.unlink()
                print(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                print(f"Could not delete {temp_file}: {e}")
    else:
        print(f"\nDeleting temporary files with name {image_name}.{extension} in {directory}")

        for temp_file in directory.glob(f"{image_name}.{extension}"):
            try:
                temp_file.unlink()  # Delete the file
                print(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                print(f"Could not delete {temp_file}: {e}")
