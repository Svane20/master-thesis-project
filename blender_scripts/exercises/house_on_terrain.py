import bpy
from mathutils import Vector

from math import radians

import numpy as np

from biomes.biomes import get_all_biomes_by_directory
from bpy_utils.bpy_data import use_backface_culling_on_materials
from constants.directories import HOUSES_DIRECTORY
from bpy_utils.bpy_ops import save_as_blend_file, render_image
from main import setup
from mesh.mesh import convert_delatin_mesh_to_sub_meshes
from mesh.mesh_generation import generate_mesh_objects_from_delation_sub_meshes
from objects.objects_creation import spawn_objects
from scene.camera import update_camera_position
from scene.light import create_light, LightType
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_terrain
from utils.utils import cleanup_directories, get_playground_directory_with_tag, move_rendered_images_to_playground

IMAGE_NAME = "house_on_terrain"

WORLD_SIZE = 100
IMAGE_SIZE = 2048

HOUSES_TO_SPAWN = 1

SEED = 42


def set_scene() -> np.ndarray:
    """
    Set up the scene

    Returns:
        The terrain.
    """

    # Add a light to the scene
    create_light(
        light_name="Sun",
        light_type=LightType.SUN,
        energy=5.0,
        location=Vector((-20, 50, 10)),
    )

    # Update camera position and rotation
    bpy.context.scene.camera = update_camera_position(
        location=Vector((-20, 40, 5)),
        rotation=Vector((radians(90), radians(0), radians(210)))
    )

    # Get all grass biomes
    grass_biomes = get_all_biomes_by_directory()

    # Create terrain and segmentation map
    terrain, segmentation_map = create_terrain_segmentation(
        world_size=WORLD_SIZE,
        image_size=IMAGE_SIZE,
        num_octaves=(1, 2),
        H=(0.0, 0.0),
        lacunarity=(0.5, 0.5),
        seed=SEED
    )

    # Create a Delatin mesh from the terrain
    delatin_mesh = create_delatin_mesh_from_terrain(terrain)

    # Convert the Delatin mesh to sub-meshes
    delatin_sub_meshes = convert_delatin_mesh_to_sub_meshes(delatin_mesh, segmentation_map)

    # Generate mesh objects from the Delatin sub-meshes
    generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes=delatin_sub_meshes,
        biomes_paths=grass_biomes,
        world_size=WORLD_SIZE,
    )

    return terrain


def main() -> None:
    # Get the playground directory
    playground_dir = get_playground_directory_with_tag(output_name=IMAGE_NAME)

    # Setup rendering engine
    setup()

    # Set up the scene
    terrain = set_scene()

    # Spawn House on the terrain
    spawn_objects(
        num_objects=HOUSES_TO_SPAWN,
        positions=np.array([[0, 0]]),
        path=HOUSES_DIRECTORY,
        terrain=terrain,
        world_size=WORLD_SIZE,
        seed=SEED
    )

    # Set backface culling for all materials
    use_backface_culling_on_materials()

    # Save the blend file
    save_as_blend_file(image_name=IMAGE_NAME)

    # Render the image
    render_image()

    # Rename the rendered image
    move_rendered_images_to_playground(playground_dir, iteration=1)

    # Cleanup the scene after rendering
    cleanup_directories()


if __name__ == "__main__":
    main()
