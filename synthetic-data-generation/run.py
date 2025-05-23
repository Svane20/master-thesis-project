import bpy
from pathlib import Path
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import time
import logging
from scipy.stats.qmc import Halton
import argparse

from bpy_utils.bpy_data import use_backface_culling_on_materials, set_scene_alpha_threshold
from bpy_utils.bpy_ops import save_as_blend_file, render_image
from addons.installation import install_addons
from configuration.configuration import Configuration, get_configuration
from configuration.spawn_objects import SpawnObjectsConfiguration
from engine.rendering import setup_rendering
from environment.biomes import get_all_biomes_by_directory
from environment.hdri import add_sky_to_scene
from environment.mesh import convert_delatin_mesh_to_sub_meshes, generate_mesh_objects_from_delation_sub_meshes, \
    populate_meshes
from environment.objects import spawn_objects
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_height_map
from environment.texture import get_all_textures_by_directory
from scene.camera import get_camera_iterations, get_random_camera_location, update_camera_position
from scene.light import create_random_light
from utils.utils import cleanup_files, get_playground_directory_with_tag, move_rendered_images_to_playground
from custom_logging.custom_logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run Blender pipeline")
    parser.add_argument('--is_colab', action='store_true', default=False,
                        help="Set if running on Colab (defaults to False)")

    return parser.parse_args()


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def general_cleanup(configuration: Configuration) -> None:
    """Perform all necessary cleanups."""

    # Remove temporary files from previous run(s)
    logging.info("Removing temporary files from previous run...")
    cleanup_files(configuration)
    logging.info("Removed temporary files.")

    # Remove the existing cube in the Blender Scene
    clear_cube()


def apply_render_configuration(configuration: Configuration) -> None:
    """Apply rendering and camera configuration in Blender."""
    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )


def initialize(configuration: Configuration) -> Configuration:
    """
    Initialization of required elements:

    - Cleanup from previous runs
    - Installs required addons
    - Setup rendering engine and outputs
    - Prepare Blender scene
    - Add a light to the scene.
    - Set the alpha threshold for the scene.

    Args:
        configuration (Configuration): The configuration for the Blender pipeline

    Returns:
        Configuration: The configuration for the Blender pipeline
    """
    logging.info("Initializing the Blender pipeline...")

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

    logging.info("Blender pipeline initialization completed.")

    return configuration


def setup_terrain(configuration: Configuration) -> NDArray[np.float32]:
    """
    Set up the terrain for the scene.

    Returns:
        NDArray[np.float32]: A height map (2D array) representing terrain.
    """
    terrain_configuration = configuration.terrain_configuration
    world_size = terrain_configuration.world_size

    logging.info("Setting up the terrain...")

    # Get all biomes
    tree_biomes = get_all_biomes_by_directory(
        directory=terrain_configuration.trees_configuration.directory,
        include=terrain_configuration.trees_configuration.include,
        exclude=terrain_configuration.trees_configuration.exclude
    )
    grass_biomes = get_all_biomes_by_directory(
        directory=terrain_configuration.grass_configuration.directory,
        include=terrain_configuration.grass_configuration.include,
        exclude=terrain_configuration.trees_configuration.exclude
    )
    not_grass_biomes = get_all_biomes_by_directory(
        directory=terrain_configuration.not_grass_configuration.directory,
        include=terrain_configuration.not_grass_configuration.include,
        exclude=terrain_configuration.trees_configuration.exclude
    )
    texture_blend_files = get_all_textures_by_directory(
        directory=terrain_configuration.textures_configuration.directory,
        include=terrain_configuration.textures_configuration.include,
        exclude=terrain_configuration.textures_configuration.exclude
    )

    # Create terrain and segmentation map
    height_map, segmentation_map = create_terrain_segmentation(
        world_size=int(world_size),
        image_size=terrain_configuration.image_size,
        noise_basis=terrain_configuration.noise_basis,
        seed=configuration.constants.seed
    )

    delatin_mesh = create_delatin_mesh_from_height_map(height_map)
    delatin_sub_meshes = convert_delatin_mesh_to_sub_meshes(delatin_mesh, segmentation_map)

    generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes=delatin_sub_meshes,
        tree_biomes_path=tree_biomes,
        grass_biomes_path=grass_biomes,
        not_grass_biomes_path=not_grass_biomes,
        generate_trees=terrain_configuration.generate_trees,
        tree_probability=terrain_configuration.tree_probability,
        world_size=int(terrain_configuration.world_size),
        seed=configuration.constants.seed
    )

    # Populate the terrain to fill empty spots in the terrain
    populate_meshes(
        delatin_mesh=delatin_mesh,
        delatin_sub_meshes=delatin_sub_meshes,
        texture_paths=texture_blend_files,
        world_size=world_size,
    )

    logging.info("Terrain setup completed.")

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
    logging.info("Spawning objects in the scene...")

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
            include=spawn_object.include,
            exclude=spawn_object.exclude,
            seed=seed
        )

    # Set backface culling for all materials
    use_backface_culling_on_materials()

    logging.info("Object spawning completed.")


def setup_the_sky(configuration: Configuration) -> None:
    """
    Set up the sky for the scene.

    Args:
        configuration (Configuration): The configuration for the scene.
    """
    logging.info("Setting up the sky...")

    add_sky_to_scene(configuration=configuration, seed=configuration.constants.seed)

    logging.info("Sky setup completed.")


def setup_scene(configuration: Configuration, start_time: float) -> NDArray[np.float32]:
    """
    Initialize the scene.

    - Generate the terrain
    - Spawn objects in the scene.
    - Generate the sky for the scene.

    Args:
        configuration (Configuration): The configuration for the Blender pipeline
        start_time (float): The start time of the script execution

    Returns:
        Tuple[Path, NDArray[np.float32]]: The configuration and the height map.
    """
    logging.info("Setting up the scene...")

    initialize(configuration)

    height_map = setup_terrain(configuration)

    spawn_objects_in_the_scene(
        configuration=configuration.spawn_objects_configuration,
        world_size=configuration.terrain_configuration.world_size,
        height_map=height_map,
        seed=configuration.constants.seed
    )

    setup_the_sky(configuration)

    # Calculate the elapsed time for setting up the scene
    elapsed_time = time.perf_counter() - start_time
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(
        f"Scene setup completed in {int(days)} days, {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds."
    )

    return height_map


def main() -> None:
    """The main function to render the images from multiple camera angles."""
    # Parse the arguments
    args = parse_args()

    # Get configuration
    configuration = get_configuration(is_colab=args.is_colab)

    # Setup logging
    setup_logging(
        name=__name__,
        log_path=configuration.run_configuration.app_path,
        save_logs=configuration.run_configuration.save_logs,
    )

    # Track the overall script execution time
    script_start_time = time.perf_counter()
    logging.info("Script execution started.")

    # Set up the scene
    height_map = setup_scene(configuration, script_start_time)

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

    try:
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
                save_as_blend_file(directory_path=str(playground_directory), iteration=index)

            # Render the image
            if configuration.constants.render_images:
                render_image(write_still=True)

            # Move rendered images to the playground directory
            move_rendered_images_to_playground(
                configuration=configuration.render_configuration.outputs_configuration,
                directory=playground_directory,
                iteration=index
            )

            # Log the elapsed time for rendering the current image
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            current_run_time = end_time - script_start_time
            elapsed_times.append(elapsed_time)  # Store elapsed time for averaging

            # Log the elapsed time for rendering the current image
            days, remainder = divmod(elapsed_time, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            logging.info(
                f"Image {current_iteration}/{total_iterations} rendered in {int(days)} days, {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds."
            )

            # Log the cumulative runtime
            days_run, run_remainder = divmod(current_run_time, 86400)
            run_hours, run_remainder = divmod(run_remainder, 3600)
            run_minutes, run_seconds = divmod(run_remainder, 60)
            logging.info(
                f"Total execution time: {int(days_run)} days, {int(run_hours)} hours, {int(run_minutes)} minutes and {run_seconds:.2f} seconds."
            )

            # Calculate remaining time estimate (except after the final iteration)
            if elapsed_times and index < total_iterations - 1:
                avg_time_per_iteration = sum(elapsed_times) / len(elapsed_times)
                remaining_iterations = total_iterations - current_iteration
                estimated_remaining_time = avg_time_per_iteration * remaining_iterations
                est_days, est_remainder = divmod(estimated_remaining_time, 86400)
                est_hours, est_remainder = divmod(est_remainder, 3600)
                est_minutes, est_seconds = divmod(est_remainder, 60)

                logging.info(
                    f"Estimated execution time remaining: {int(est_days)} days, {int(est_hours)} hours, {int(est_minutes)} minutes and {est_seconds:.2f} seconds."
                )
    except KeyboardInterrupt:
        logging.error("Keyboard interrupt detected. Terminating the script.")

        # Delete playground directory if empty
        if not list(playground_directory.iterdir()):
            logging.info(f"Deleting empty playground directory: {playground_directory}")
            playground_directory.rmdir()
    except Exception as e:
        logging.exception(f"An error occurred during the script execution: Error: {e}")

        # Delete playground directory if empty
        if not list(playground_directory.iterdir()):
            logging.info(f"Deleting empty playground directory: {playground_directory}")
            playground_directory.rmdir()

    # Cleanup temporary files generated during rendering
    logging.info("Removing temporary files generated during the execution...")
    cleanup_files(configuration)
    logging.info("Removed temporary files.")

    # Log the total execution time of the script
    script_end_time = time.perf_counter()
    total_elapsed_time = script_end_time - script_start_time
    total_days, total_remainder = divmod(total_elapsed_time, 86400)
    total_hours, total_remainder = divmod(total_remainder, 3600)
    total_minutes, total_seconds = divmod(total_remainder, 60)
    logging.info(
        f"Script finished in {int(total_days)} days, {int(total_hours)} hours, {int(total_minutes)} minutes and {total_seconds:.2f} seconds."
    )


if __name__ == "__main__":
    main()
