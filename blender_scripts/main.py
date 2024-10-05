import bpy

from configuration.addons import install_addons
from configuration.consts import Constants
from engine.rendering import setup_rendering
from configuration.configuration import Configuration, save_configuration, load_configuration
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def setup() -> None:
    clear_cube()

    # Install addons
    install_addons()

    # Load settings
    config = load_configuration(path=Constants.Directory.CONFIG_PATH)
    if config is None:
        config = save_configuration(configuration=Configuration().model_dump(), path=Constants.Directory.CONFIG_PATH)

    configuration = Configuration(**config)

    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )


def main() -> None:
    setup()


if __name__ == "__main__":
    main()
