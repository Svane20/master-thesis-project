import bpy

from pathlib import Path

from consts import Constants


class BpyOpsConstants:
    IMAGE_EXTENSION: str = "png"
    IMAGE_FORMAT: str = "PNG"
    IMAGE_COLOR_MODE: str = "RGBA"


def render_image(image_name: str, directory_path: Path = Constants.Directory.OUTPUT_DIR,
                 write_still: bool = True) -> None:
    """Saves the current render as an image file."""
    try:
        # Ensure the directory exists
        output_dir: Path = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Set the render settings
        scene: bpy.types.Scene = bpy.context.scene
        render: bpy.types.RenderSettings = scene.render

        # Set the output filepath
        render_path = str(output_dir / f"{image_name}.{BpyOpsConstants.IMAGE_EXTENSION}")
        render.filepath = render_path

        # Set the image format and color mode
        render.image_settings.file_format = BpyOpsConstants.IMAGE_FORMAT
        render.image_settings.color_mode = BpyOpsConstants.IMAGE_COLOR_MODE

        # Render the image
        bpy.ops.render.render(write_still=write_still)
    except Exception as e:
        print(f"Failed to save image file: {e}")


def save_as_blend_file(image_name: str, directory_path: Path = Constants.Directory.BLENDER_FILES_DIR, allow_overwrite: bool = True) -> None:
    """Saves the current Blender scene as a .blend file."""
    try:
        # Ensure the directory exists
        output_dir: Path = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_name}.blend"

        # Remove the existing file if it exists
        if allow_overwrite and output_path.exists():
            output_path.unlink()

        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        print(f"Saved: '{output_path}'")
    except Exception as e:
        print(f"Failed to save blend file: {e}")
