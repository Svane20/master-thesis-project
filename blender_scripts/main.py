import bpy

from configuration.addons import install_addons
from engine.rendering import setup_rendering
from configuration.configuration import Configuration, save_configuration, load_configuration
from custom_logging.custom_logger import setup_logger
from utils.utils import cleanup_directories

logger = setup_logger(__name__)


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def setup() -> None:
    """Set up the scene for rendering"""

    # Cleanup the scene before rendering
    cleanup_directories(remove_blender_dir=True)

    # Clear the cube object if it exists
    clear_cube()

    # Install the required addons
    install_addons()

    config = load_configuration()
    if config is None:
        config = save_configuration(configuration=Configuration().model_dump())

    configuration = Configuration(**config)

    # Set up Blender rendering configuration
    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )


def main() -> None:
    setup()


if __name__ == "__main__":
    main()
