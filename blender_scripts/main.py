import bpy

from consts import Constants
from rendering.rendering import setup_rendering, set_noise_threshold
from configs.configuration import Configuration, save_configuration, load_configuration
from utils import save_blend_file


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")

    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def setup(output_name: str = None) -> None:
    clear_cube()

    # Load settings
    config = load_configuration(path=Constants.Directory.CONFIG_PATH)
    if config is None:
        config = save_configuration(configuration=Configuration().model_dump(), path=Constants.Directory.CONFIG_PATH)

    configuration = Configuration(**config)
    terrain_configuration = configuration.terrain_configuration

    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
        output_name=output_name,
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
