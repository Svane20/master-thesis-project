import bpy
from mathutils import Vector

from math import radians

from biomes.biomes import get_all_biomes_by_directory
from engine.bpy_ops import save_as_blend_file, render_image
from main import setup
from mesh.mesh import convert_delatin_mesh_to_sub_meshes
from mesh.mesh_generation import generate_mesh_objects_from_delation_sub_meshes
from scene.camera import update_camera_position
from scene.light import create_light, LightType
from environment.terrain import create_terrain_segmentation, create_delatin_mesh_from_terrain
from utils.utils import cleanup_directories, get_playground_directory_with_tag, move_rendered_images_to_playground

IMAGE_NAME = "terrain_generation"

WORLD_SIZE = 30
IMAGE_SIZE = 2048

NUM_OCTAVES = (1, 2)
H = (0.0, 0.0)
LACUNARITY = (0.5, 0.5)
SEED = 42


def set_scene() -> None:
    """Set up the scene."""

    # Add a light to the scene
    create_light(
        light_name="Sun",
        light_type=LightType.SUN,
        energy=5.0,
    )

    # Update camera position and rotation
    bpy.context.scene.camera = update_camera_position(
        location=Vector((30, 0, 30)),
        rotation=Vector((radians(40), radians(0), radians(90)))
    )


def main() -> None:
    # Get the playground directory
    playground_dir = get_playground_directory_with_tag(output_name=IMAGE_NAME)

    # Setup rendering engine
    setup()

    # Set up the scene
    set_scene()

    # Get all grass biomes
    grass_biomes = get_all_biomes_by_directory()

    terrain, segmentation_map = create_terrain_segmentation(
        world_size=WORLD_SIZE,
        image_size=IMAGE_SIZE,
        num_octaves=NUM_OCTAVES,
        H=H,
        lacunarity=LACUNARITY,
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
