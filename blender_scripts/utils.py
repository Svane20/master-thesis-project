import shutil
from subprocess import CalledProcessError, CompletedProcess, run, PIPE

from typing import Union
from pathlib import Path

from configs.configuration import RenderConfiguration
from consts import Constants

BLENDER_EXECUTABLE = shutil.which("blender")
if BLENDER_EXECUTABLE is None:
    print("Blender executable not found")
    raise SystemExit("Blender executable not found. Ensure Blender is installed and accessible in the system PATH.")


def open_blend_file_in_blender(blender_file: Path) -> None:
    """Opens a .blend file."""
    try:
        result: CompletedProcess[str] = run(
            [
                BLENDER_EXECUTABLE,
                blender_file,
            ],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            check=True
        )

        print(f"Blender rendering completed successfully. Output: {result.stdout}")
        if result.stderr:
            print(f"Blender rendering errors: {result.stderr}")
    except CalledProcessError as e:
        print(f"Blender rendering failed: {e.stderr}")
    except Exception as e:
        print(f"An error occurred while running the Blender process: {e}")


def run_blender_file(blender_file: Path) -> None:
    """Runs a .blend file."""
    try:
        result = run(
            [
                BLENDER_EXECUTABLE,
                "--background",
                blender_file,
                "--render-output",
                f"{Constants.Directory.OUTPUT_DIR}/output",
                "--render-format",
                "PNG",
                "--use-extension",
                "1",
                "--render-frame",
                "1",
            ],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            check=True
        )

        print(f"Blender rendering completed successfully. Output: {result.stdout}")
        if result.stderr:
            print(f"Blender rendering errors: {result.stderr}")
    except CalledProcessError as e:
        print(f"Blender rendering failed: {e.stderr}")
    except Exception as e:
        print(f"An error occurred while running the Blender process: {e}")


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
