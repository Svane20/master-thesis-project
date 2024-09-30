import bpy

from pathlib import Path
import logging

from consts import Constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def render_image(
        image_name: str,
        directory_path: Path = Constants.Directory.OUTPUT_DIR,
        write_still: bool = True
) -> None:
    """
    Renders the current scene as an image.

    Args:
        image_name: The name of the image.
        directory_path: The directory path.
        write_still: Whether to write the image to disk.

    Raises:
        Exception: If the image fails to render.
    """
    try:
        output_dir = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        scene = bpy.context.scene
        render = scene.render

        render.filepath = str(output_dir / f"{image_name}.{Constants.FileExtension.PNG}")
        render.image_settings.file_format = Constants.Render.FILE_FORMAT
        render.image_settings.color_mode = Constants.Render.COLOR_MODE

        bpy.ops.render.render(write_still=write_still)

        logger.info(f"Rendered: '{render.filepath}'")
    except Exception as e:
        logger.error(f"Failed to render image: {e}")
        raise


def save_as_blend_file(
        image_name: str,
        directory_path: Path = Constants.Directory.BLENDER_FILES_DIR,
        allow_overwrite: bool = True
) -> None:
    """
    Saves the current Blender scene as a .blend file.

    Args:
        image_name: The name of the image.
        directory_path: The directory path.
        allow_overwrite: Whether to allow overwriting the file.

    Raises:
        Exception: If the blend file fails to save.
    """
    try:
        # Ensure the directory exists
        output_dir: Path = Path(directory_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_name}.{Constants.FileExtension.BLEND}"

        # Remove the existing file if it exists
        if allow_overwrite and output_path.exists():
            output_path.unlink()

        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        logger.info(f"Saved: '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save blend file: {e}")
        raise
