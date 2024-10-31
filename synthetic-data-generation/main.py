import bpy

from pathlib import Path
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from bpy_utils.bpy_data import use_backface_culling_on_materials, set_scene_alpha_threshold
from bpy_utils.bpy_ops import save_as_blend_file, render_image
from configuration.addons import install_addons
from configuration.configuration import Configuration, load_configuration, save_configuration
from configuration.hdri import HDRIConfiguration
from constants.directories import HOUSES_DIRECTORY
from engine.rendering import setup_rendering
from environment.biomes import get_all_biomes_by_directory
from environment.hdri import add_sky_to_scene
from environment.mesh import convert_delatin_mesh_to_sub_meshes, generate_mesh_objects_from_delation_sub_meshes
from environment.objects import spawn_objects
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_height_map
from scene.camera import get_camera_iterations, get_random_camera_location, update_camera_position
from scene.light import create_random_light
from utils.utils import cleanup_files, get_playground_directory_with_tag, move_rendered_images_to_playground
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

IMAGE_NAME = "full_example"
WORLD_SIZE = 100
IMAGE_SIZE = 2048
SEED = 1


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


def initialize() -> Tuple[Path, NDArray[np.float64]]:
    """
    Initialization of required elements:

    - Cleanup from previous runs
    - Installs required addons
    - Setup rendering engine and outputs
    - Prepare Blender scene
    - Add a light to the scene.
    - Set the alpha threshold for the scene.
    - Generate the playground directory.
    - Get the camera iterations.

    Returns:
        Tuple[Path, NDArray[np.float64]]: The playground directory and camera iterations.
    """
    # General cleanup function to handle all object and directory cleanups
    general_cleanup()

    # Install necessary Blender addons
    install_addons()

    # Handle configuration setup
    configuration = load_and_save_configuration()

    # Apply the rendering setup
    apply_render_configuration(configuration)

    create_random_light(
        light_name="Sun",
        seed=SEED,
    )

    set_scene_alpha_threshold(alpha_threshold=0.5)

    playground_directory = get_playground_directory_with_tag(output_name=IMAGE_NAME)

    iterations = get_camera_iterations(seed=SEED)

    return playground_directory, iterations


def setup_terrain() -> NDArray[np.float32]:
    """
    Set up the terrain for the scene.

    Returns:
        NDArray[np.float32]: A height map (2D array) representing terrain.
    """
    grass_biomes = get_all_biomes_by_directory()

    # Create terrain and segmentation map
    height_map, segmentation_map = create_terrain_segmentation(
        world_size=WORLD_SIZE,
        image_size=IMAGE_SIZE,
        num_octaves=(1, 2),
        H=(0.0, 0.0),
        lacunarity=(0.5, 0.5),
        seed=SEED
    )

    delatin_mesh = create_delatin_mesh_from_height_map(height_map)
    delatin_sub_meshes = convert_delatin_mesh_to_sub_meshes(delatin_mesh, segmentation_map)

    generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes=delatin_sub_meshes,
        biomes_paths=grass_biomes,
        world_size=WORLD_SIZE,
    )

    return height_map


def spawn_objects_in_the_scene(height_map: NDArray[np.float32]) -> None:
    """
    Spawn objects in the scene.

    Args:
        height_map (NDArray[np.float32]): The terrain height map.
    """

    # Spawn House on the terrain
    spawn_objects(
        num_objects=1,
        positions=np.array([[0, 0]]),
        path=HOUSES_DIRECTORY,
        height_map=height_map,
        world_size=WORLD_SIZE,
        seed=SEED
    )

    # Set backface culling for all materials
    use_backface_culling_on_materials()


def setup_the_sky(configuration: HDRIConfiguration) -> None:
    """
    Set up the sky for the scene.

    Args:
        configuration (HDRIConfiguration): The HDRI configuration.
    """
    add_sky_to_scene(configuration=configuration, seed=SEED)


def setup_scene() -> Tuple[Path, NDArray[np.float32], NDArray[np.float64]]:
    """
    Initialize the scene.

    Returns:
        Tuple[Path, NDArray[np.float32], NDArray[np.float64]]: The playground directory, the height map and camera iterations.
    """
    playground_directory, iterations = initialize()

    height_map = setup_terrain()

    spawn_objects_in_the_scene(height_map)

    setup_the_sky(configuration=HDRIConfiguration())

    return playground_directory, height_map, iterations


def main() -> None:
    """The main function to render the images from multiple camera angles."""
    # Set up the scene
    playground_directory, height_map, iterations = setup_scene()

    # Get the total number of iterations
    total_iterations = len(iterations)
    logger.info(f"Total iterations: {total_iterations}")

    # Render images from multiple camera angles
    for index, iteration in enumerate(iterations):
        current_iteration = index + 1
        logger.info(f"Rendering image {current_iteration}/{total_iterations}")

        location = get_random_camera_location(
            iteration=iteration,
            height_map=height_map,
            world_size=WORLD_SIZE,
            seed=SEED
        )

        update_camera_position(location=location)

        if index == 0:
            save_as_blend_file(image_name=IMAGE_NAME)

        render_image(write_still=True)

        # Rename the rendered image and mask(s)
        move_rendered_images_to_playground(directory=playground_directory, iteration=index)

        logger.info(f"Image {current_iteration}/{total_iterations} rendered successfully")

    # Cleanup temporary files generated during rendering
    cleanup_files()


if __name__ == "__main__":
    main()
