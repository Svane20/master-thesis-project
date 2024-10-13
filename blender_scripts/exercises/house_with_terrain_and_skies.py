import bpy
from mathutils import Vector

from math import radians

import numpy as np

from environment.biomes import get_all_biomes_by_directory
from bpy_utils.bpy_data import use_backface_culling_on_materials, set_scene_alpha_threshold
from constants.directories import HOUSES_DIRECTORY
from bpy_utils.bpy_ops import save_as_blend_file, render_image
from environment.hdri import add_sky_to_scene
from environment.objects import spawn_objects
from main import setup
from environment.mesh import convert_delatin_mesh_to_sub_meshes, generate_mesh_objects_from_delation_sub_meshes
from scene.camera import update_camera_position
from scene.light import create_light, LightType
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_terrain
from utils.utils import cleanup_files, get_playground_directory_with_tag, move_rendered_images_to_playground

IMAGE_NAME = "house_with_terrain_and_skies"

SEED = 42


def set_scene() -> None:
    """Set up the scene"""
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
        world_size=100,
        image_size=2048,
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
        world_size=100,
    )

    # Spawn House on the terrain
    spawn_objects(
        num_objects=1,
        positions=np.array([[0, 0]]),
        path=HOUSES_DIRECTORY,
        terrain=terrain,
        world_size=100,
        seed=SEED
    )

    # Set backface culling for all materials
    use_backface_culling_on_materials()

    # Set the alpha threshold for the scene
    set_scene_alpha_threshold(alpha_threshold=0.5)


def main() -> None:
    # Get the playground directory
    playground_dir = get_playground_directory_with_tag(output_name=IMAGE_NAME)

    # Setup rendering engine
    configuration = setup()

    # Set up the scene
    set_scene()

    # Add sky to the scene
    add_sky_to_scene(configuration=configuration.hdri_configuration, seed=SEED)

    # Save the blend file
    save_as_blend_file(image_name=IMAGE_NAME)

    # Render the image
    render_image()

    # Rename the rendered image
    move_rendered_images_to_playground(playground_dir, iteration=1)

    # Cleanup the scene after rendering
    cleanup_files()


if __name__ == "__main__":
    main()
