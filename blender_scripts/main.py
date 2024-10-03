import bpy

import logging

from addons import install_addons
from consts import Constants
from rendering.rendering import setup_rendering
from configs.configuration import Configuration, save_configuration, load_configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def install_dependencies() -> None:
    install_addons()


def setup() -> None:
    clear_cube()

    # Install dependencies
    install_dependencies()

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
