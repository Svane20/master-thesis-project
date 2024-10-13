import bpy

from configuration.addons import install_addons
from configuration.configuration import Configuration, load_configuration, save_configuration
from engine.rendering import setup_rendering

from custom_logging.custom_logger import setup_logger
from utils.utils import cleanup_files

logger = setup_logger(__name__)


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def general_cleanup() -> None:
    """Perform all necessary cleanups."""
    cleanup_files(remove_blender_dir=True)
    clear_cube()


def load_and_save_configuration() -> Configuration:
    """Load configuration and save it for future use."""
    save_configuration(Configuration().model_dump())

    config = load_configuration()
    configuration = Configuration(**config)

    return configuration


def apply_render_configuration(configuration: Configuration) -> None:
    """Apply rendering and camera configuration in Blender."""
    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )


def setup_scene() -> None:
    """Prepare the Blender scene and configuration for rendering."""
    # General cleanup function to handle all object and directory cleanups
    general_cleanup()

    # Install necessary Blender addons
    install_addons()

    # Handle configuration setup
    configuration = load_and_save_configuration()

    # Apply the rendering setup
    apply_render_configuration(configuration)


def setup() -> Configuration:
    """
    Set up the scene for rendering

    Returns:
        Configuration: The configuration object for the scene.
    """

    # Cleanup the scene before rendering
    cleanup_files(remove_blender_dir=True)

    # Clear the cube object if it exists
    clear_cube()

    # Install the required addons
    install_addons()

    # Save the configuration
    save_configuration(configuration=Configuration().model_dump())

    # Load the configuration
    config = load_configuration()
    configuration = Configuration(**config)

    # Set up Blender rendering configuration
    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )

    return configuration


def main() -> None:
    setup_scene()


if __name__ == "__main__":
    main()
