from typing import Union

import bpy
import shutil
import subprocess

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
        result = subprocess.run(
            [
                BLENDER_EXECUTABLE,
                blender_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print(f"Blender rendering completed successfully. Output: {result.stdout}")
        if result.stderr:
            print(f"Blender rendering errors: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Blender rendering failed: {e.stderr}")
    except Exception as e:
        print(f"An error occurred while running the Blender process: {e}")


def save_blend_file(path: Path) -> None:
    """Saves the current Blender scene as a .blend file."""
    try:
        # Ensure the directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bpy.ops.wm.save_as_mainfile(filepath=str(path))
    except Exception as e:
        print(f"Failed to save blend file: {e}")
    finally:
        bpy.ops.wm.read_factory_settings(use_empty=True)


def save_image_file(directory_path: Path = Constants.Directory.OUTPUT_DIR) -> None:
    """Saves the current render as an image file."""
    try:
        # Ensure the directory exists
        directory_path = Path(directory_path)
        directory_path.parent.mkdir(parents=True, exist_ok=True)

        # Render and save the image
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print(f"Failed to save image file: {e}")
    finally:
        bpy.ops.wm.read_factory_settings(use_empty=True)


def get_temporary_file_path(file_name: Union[str | None], render_configuration: RenderConfiguration) -> str:
    temp_dir = Path(render_configuration.tmp_folder)

    path = temp_dir / (file_name or "temp")
    path.parent.mkdir(parents=True, exist_ok=True)

    return path.as_posix()
