import bpy

from pathlib import Path

from consts import Constants
from utils import remove_temporary_files


def render_image(directory_path: Path = Constants.Directory.OUTPUT_DIR, write_still: bool = True) -> None:
    """Saves the current render as an image file."""
    try:
        # Ensure the directory exists
        output_dir: Path = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Render and save the image
        bpy.ops.render.render(write_still=write_still)
    except Exception as e:
        print(f"Failed to save image file: {e}")


def save_as_blend_file(image_name: str, directory_path: Path = Constants.Directory.BLENDER_FILES_DIR) -> None:
    """Saves the current Blender scene as a .blend file."""
    try:
        # Ensure the directory exists
        output_dir: Path = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Save the blend file
        output_path = output_dir / f"{image_name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        print(f"Saved: '{output_path}'")

        # Remove temporary files
        remove_temporary_files(output_dir, image_name, extension="blend1")
    except Exception as e:
        print(f"Failed to save blend file: {e}")
