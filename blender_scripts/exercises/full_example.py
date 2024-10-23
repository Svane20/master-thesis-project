from pathlib import Path
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from bpy_utils.bpy_data import use_backface_culling_on_materials, set_scene_alpha_threshold
from bpy_utils.bpy_ops import render_image, save_as_blend_file
from configuration.hdri import HDRIConfiguration
from constants.directories import HOUSES_DIRECTORY
from custom_logging.custom_logger import setup_logger
from environment.biomes import get_all_biomes_by_directory
from environment.hdri import add_sky_to_scene
from environment.mesh import convert_delatin_mesh_to_sub_meshes, generate_mesh_objects_from_delation_sub_meshes
from environment.objects import spawn_objects
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_height_map
from main import setup
from scene.camera import get_camera_iterations, get_random_camera_location, update_camera_position
from scene.light import create_random_light
from utils.utils import get_playground_directory_with_tag, move_rendered_images_to_playground, cleanup_files

logger = setup_logger(__name__)

IMAGE_NAME = "full_example"
WORLD_SIZE = 100
IMAGE_SIZE = 2048
SEED = 1


def setup_terrain() -> NDArray[np.float32]:
    """
    Set up the terrain for the scene.

    Returns:
        NDArray[np.float32]: A height map (2D array) representing terrain.
    """

    # Get all grass biomes
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

    # Create a Delatin mesh from the terrain
    delatin_mesh = create_delatin_mesh_from_height_map(height_map)

    # Convert the Delatin mesh to sub-meshes
    delatin_sub_meshes = convert_delatin_mesh_to_sub_meshes(delatin_mesh, segmentation_map)

    # Generate mesh objects from the Delatin sub-meshes
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
    # Add sky to the scene
    add_sky_to_scene(configuration=configuration, seed=SEED)


def setup_the_scene() -> Tuple[Path, NDArray[np.float64]]:
    """
    Set up the scene:

    - Add a light to the scene.
    - Set the alpha threshold for the scene.
    - Generate the playground directory.
    - Get the camera iterations.

    Returns:
        Tuple[Path, NDArray[np.float64]]: The playground directory and camera iterations.
    """
    # Add a light to the scene
    create_random_light(
        light_name="Sun",
        seed=SEED,
    )

    # Set the alpha threshold for the scene
    set_scene_alpha_threshold(alpha_threshold=0.5)

    # Get the playground directory
    playground_directory = get_playground_directory_with_tag(output_name=IMAGE_NAME)

    # Get the camera iterations
    iterations = get_camera_iterations(seed=SEED)

    return playground_directory, iterations


def initialize() -> Tuple[Path, NDArray[np.float32], NDArray[np.float64]]:
    """
    Initialize the scene.

    Returns:
        Tuple[Path, NDArray[np.float32], NDArray[np.float64]]: The playground directory, the height map and camera iterations.
    """
    # Setup rendering engine
    setup()

    # Set up the terrain
    height_map = setup_terrain()

    # Spawn objects in the scene
    spawn_objects_in_the_scene(height_map)

    # Set up the sky
    setup_the_sky(configuration=HDRIConfiguration())

    # Set up the scene
    playground_directory, iterations = setup_the_scene()

    return playground_directory, height_map, iterations


def main() -> None:
    # Set up the scene
    playground_directory, height_map, iterations = initialize()

    # Get the total number of iterations
    total_iterations = len(iterations)
    logger.info(f"Total iterations: {total_iterations}")

    # Render images from multiple camera angles
    for index, iteration in enumerate(iterations):
        current_iteration = index + 1
        logger.info(f"Rendering image {current_iteration}/{total_iterations}")

        # Get random camera location
        location = get_random_camera_location(
            iteration=iteration,
            height_map=height_map,
            world_size=WORLD_SIZE,
            seed=SEED
        )

        # Update camera location
        update_camera_position(location=location)

        # Save the blend file
        if index == 0:
            save_as_blend_file(image_name=IMAGE_NAME)

        # Render the image
        render_image(write_still=True)

        # Rename the rendered image and masks
        move_rendered_images_to_playground(directory=playground_directory, iteration=index)

        logger.info(f"Image {current_iteration}/{total_iterations} rendered successfully")

    # Cleanup temporary files generated during rendering
    cleanup_files()


if __name__ == "__main__":
    main()
