import bpy
from pathlib import Path
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import time
import logging
import platform
from scipy.stats.qmc import Halton

from bpy_utils.bpy_data import use_backface_culling_on_materials, set_scene_alpha_threshold
from bpy_utils.bpy_ops import save_as_blend_file, render_image
from addons.installation import install_addons
from configuration.configuration import Configuration, load_configuration
from configuration.spawn_objects import SpawnObjectsConfiguration
from engine.rendering import setup_rendering
from environment.biomes import get_all_biomes_by_directory
from environment.hdri import add_sky_to_scene
from environment.mesh import convert_delatin_mesh_to_sub_meshes, generate_mesh_objects_from_delation_sub_meshes, \
    populate_meshes
from environment.objects import spawn_objects
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_height_map
from scene.camera import get_camera_iterations, get_random_camera_location, update_camera_position
from scene.light import create_random_light
from utils.utils import cleanup_files, get_playground_directory_with_tag, move_rendered_images_to_playground
from custom_logging.custom_logger import setup_logging


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def general_cleanup(configuration: Configuration) -> None:
    """Perform all necessary cleanups."""
    cleanup_files(configuration)
    clear_cube()


def get_configuration() -> Configuration:
    """
    Loads the configuration for the Blender pipeline.

    Returns:
        Configuration: The configuration for the Blender pipeline

    """
    # Detect OS and set the configuration path accordingly
    base_directory = Path(__file__).resolve().parent
    if platform.system() == "Windows":
        configuration_path: Path = base_directory / "configuration_windows.json"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = base_directory / "configuration_linux.json"

    config = load_configuration(configuration_path)

    return Configuration(**config)


def apply_render_configuration(configuration: Configuration) -> None:
    """Apply rendering and camera configuration in Blender."""
    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )


def initialize() -> Configuration:
    """
    Initialization of required elements:

    - Cleanup from previous runs
    - Installs required addons
    - Setup rendering engine and outputs
    - Prepare Blender scene
    - Add a light to the scene.
    - Set the alpha threshold for the scene.

    Returns:
        Configuration: The configuration for the Blender pipeline
    """
    # Handle configuration setup
    configuration = get_configuration()

    # General cleanup function to handle all object and directory cleanups
    general_cleanup(configuration)

    # Install necessary Blender addons
    install_addons(configuration.addons)

    # Apply the rendering setup
    apply_render_configuration(configuration)

    # Create light
    create_random_light(
        light_name="Sun",
        seed=configuration.constants.seed,
    )

    # Set the alpha threshold of the scene
    set_scene_alpha_threshold(alpha_threshold=0.5)

    return configuration


def setup_terrain(configuration: Configuration) -> NDArray[np.float32]:
    """
    Set up the terrain for the scene.

    Returns:
        NDArray[np.float32]: A height map (2D array) representing terrain.
    """
    terrain_configuration = configuration.terrain_configuration
    world_size = terrain_configuration.world_size

    # Get all biomes
    tree_biomes = get_all_biomes_by_directory(
        directory=terrain_configuration.trees_configuration.directory,
        keywords=terrain_configuration.trees_configuration.keywords,
    )
    grass_biomes = get_all_biomes_by_directory(
        directory=terrain_configuration.grass_configuration.directory,
        keywords=terrain_configuration.grass_configuration.keywords
    )
    texture_blend_paths = [
        file_path
        for directory in terrain_configuration.textures_configuration.directories
        for file_path in Path(directory).rglob("*.blend")
        if str(file_path).endswith(".blend")
    ]

    # Create terrain and segmentation map
    height_map, segmentation_map = create_terrain_segmentation(
        world_size=int(world_size),
        image_size=terrain_configuration.image_size,
        noise_basis=terrain_configuration.noise_basis,
        num_octaves=(1, 2),
        H=(0.0, 0.0),
        lacunarity=(0.5, 0.5),
        seed=1  # Ensure we always generate the same terrain so we don't get white spots in the grass
    )

    delatin_mesh = create_delatin_mesh_from_height_map(height_map)
    delatin_sub_meshes = convert_delatin_mesh_to_sub_meshes(delatin_mesh, segmentation_map)

    generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes=delatin_sub_meshes,
        tree_biomes_path=tree_biomes,
        grass_biomes_path=grass_biomes,
        generate_trees=terrain_configuration.generate_trees,
        world_size=int(terrain_configuration.world_size),
        seed=configuration.constants.seed
    )

    # Populate the terrain to fill empty spots in the terrain
    populate_meshes(
        delatin_mesh=delatin_mesh,
        delatin_sub_meshes=delatin_sub_meshes,
        texture_paths=texture_blend_paths,
        world_size=world_size,
    )

    return height_map


def spawn_objects_in_the_scene(
        configuration: SpawnObjectsConfiguration,
        world_size: float,
        height_map: NDArray[np.float32],
        seed: int = None
) -> None:
    """
    Spawn objects in the scene.

    Args:
        configuration (Configuration): The configuration for the scene.
        world_size (float): The world size of the scene.
        height_map (NDArray[np.float32]): The terrain height map.
        seed (int, optional): The seed for the random number generator.
    """
    # Spawn objects on the terrain
    for spawn_object in configuration.spawn_objects:
        if not spawn_object.should_spawn:
            continue

        if spawn_object.use_halton:
            halton = Halton(d=2)
            position = ((halton.random(n=30) - 0.5) * world_size).reshape(-1, 2)
        else:
            assert spawn_object.position is not None, "Spawn location is not set for spawn object"
            position = np.array([spawn_object.position])

        spawn_objects(
            num_objects=spawn_object.num_objects,
            positions=position,
            filepath=spawn_object.directory,
            height_map=height_map,
            world_size=world_size,
            seed=seed
        )

    # Set backface culling for all materials
    use_backface_culling_on_materials()


def setup_the_sky(configuration: Configuration) -> None:
    """
    Set up the sky for the scene.

    Args:
        configuration (Configuration): The configuration for the scene.
    """
    add_sky_to_scene(configuration=configuration, seed=configuration.constants.seed)


def setup_scene() -> Tuple[Configuration, NDArray[np.float32]]:
    """
    Initialize the scene.

    - Generate the terrain
    - Spawn objects in the scene.
    - Generate the sky for the scene.

    Returns:
        Tuple[Path, NDArray[np.float32]]: The configuration and the height map.
    """
    configuration = initialize()

    height_map = setup_terrain(configuration)

    spawn_objects_in_the_scene(
        configuration=configuration.spawn_objects_configuration,
        world_size=configuration.terrain_configuration.world_size,
        height_map=height_map,
        seed=configuration.constants.seed
    )

    setup_the_sky(configuration)

    return configuration, height_map


def main() -> None:
    """The main function to render the images from multiple camera angles."""
    setup_logging(__name__)

    # Track the overall script execution time
    script_start_time = time.perf_counter()
    logging.info("Script execution started.")

    # Set up the scene
    configuration, height_map = setup_scene()

    # Create output directory and set camera iterations
    playground_directory = get_playground_directory_with_tag(configuration=configuration)
    iterations = get_camera_iterations(
        num_iterations=configuration.constants.num_iterations,
        seed=configuration.constants.seed
    )
    total_iterations = len(iterations)
    logging.info(f"Total image generation iterations: {total_iterations}")

    # Track execution times for estimation
    elapsed_times = []

    # Render images from multiple camera angles
    for index, iteration in enumerate(iterations):
        current_iteration = index + 1
        logging.info(f"Rendering image {current_iteration}/{total_iterations}")
        start_time = time.perf_counter()

        location = get_random_camera_location(
            iteration=iteration,
            height_map=height_map,
            world_size=int(configuration.terrain_configuration.world_size),
            seed=configuration.constants.seed
        )

        update_camera_position(location=location)

        # Save the Blender file
        if configuration.constants.save_blend_files:
            save_as_blend_file(iteration=index, directory_path=str(playground_directory))

        # Render the image
        if configuration.constants.render_images:
            render_image(write_still=True)

            # Rename the rendered image and mask(s)
            move_rendered_images_to_playground(
                configuration=configuration.render_configuration.outputs_configuration,
                directory=playground_directory,
                iteration=index
            )

        # Log the elapsed time for rendering the current image
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)  # Store elapsed time for averaging

        minutes, seconds = divmod(elapsed_time, 60)
        logging.info(
            f"Image {current_iteration}/{total_iterations} rendered successfully "
            f"(Execution time: {int(minutes)} minutes and {seconds:.2f} seconds)"
        )

        # Calculate remaining time estimate
        if elapsed_times:
            avg_time_per_iteration = sum(elapsed_times) / len(elapsed_times)
            remaining_iterations = total_iterations - current_iteration
            estimated_remaining_time = avg_time_per_iteration * remaining_iterations
            est_minutes, est_seconds = divmod(estimated_remaining_time, 60)

            logging.info(
                f"Estimated time remaining: {int(est_minutes)} minutes and {est_seconds:.2f} seconds."
            )

    # Cleanup temporary files generated during rendering
    cleanup_files(configuration)
    logging.info("Temporary files cleaned up.")

    # Log the total execution time of the script
    script_end_time = time.perf_counter()
    total_elapsed_time = script_end_time - script_start_time
    total_minutes, total_seconds = divmod(total_elapsed_time, 60)
    logging.info(f"Script finished in {int(total_minutes)} minutes and {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
