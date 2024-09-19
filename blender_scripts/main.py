from pathlib import Path

import bpy

from consts import Constants
from rendering.rendering import setup_rendering, set_noise_threshold
from configs.configuration import Configuration, save_configuration, load_configuration
from utils import save_blend_file


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    if bpy.data.objects.get("Cube"):
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()


def setup(output_dir: Path = Constants.Directory.OUTPUT_DIR) -> None:
    clear_cube()

    # Save the existing settings
    configuration = Configuration()
    save_configuration(configuration=configuration.model_dump(), path=Constants.Directory.CONFIG_PATH)

    # Load settings
    config = load_configuration(path=Constants.Directory.CONFIG_PATH)
    configuration = Configuration(**config)
    terrain_configuration = configuration.terrain_configuration

    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
        output_dir=output_dir,
        render_object_index=True,
        world_size=int(terrain_configuration.world_size)
    )
    set_noise_threshold(Constants.Default.NOISE_THRESHOLD)


def main() -> None:
    setup()

    save_blend_file(path=Constants.Directory.BLENDER_FILES_PATH)
    # open_blend_file_in_blender(blender_file=Constants.Directory.BLENDER_FILES_PATH)


if __name__ == "__main__":
    main()
