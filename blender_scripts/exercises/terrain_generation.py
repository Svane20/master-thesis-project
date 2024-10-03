import bpy

import numpy as np
from typing import List

from biomes import get_all_biomes, apply_biome
from bpy_ops import save_as_blend_file
from main import setup
from mesh import convert_delatin_mesh_to_sub_meshes
from terrain import create_terrain_segmentation, create_delatin_mesh_from_terrain
from visualization import visualize_terrain_sub_meshes_3d, visualize_terrain_sub_meshes_2d, visualize_terrain, \
    visualize_terrain_mesh

IMAGE_NAME = "terrain_generation"

WORLD_SIZE = 20
IMAGE_SIZE = 2048


def get_unique_object_names(existing_object_names: List[str], new_object_names: List[str]) -> set[str]:
    return set(new_object_names) - set(existing_object_names)


def main() -> None:
    setup()

    grass_biomes = get_all_biomes()

    terrain, segmentation_map, (grass, texture, beds) = create_terrain_segmentation(
        world_size=WORLD_SIZE,
        image_size=IMAGE_SIZE,
        num_octaves=(1, 2),
        H=(0.1, 0.2),
        lacunarity=(0.5, 0.8),
        seed=42
    )
    delatin_mesh = create_delatin_mesh_from_terrain(terrain)
    delatin_sub_meshes = convert_delatin_mesh_to_sub_meshes(delatin_mesh, segmentation_map)

    # visualize_terrain(terrain, segmentation_map, grass, texture, beds)
    # visualize_terrain_mesh(delatin_mesh)
    # visualize_terrain_sub_meshes_2d(delatin_sub_meshes)
    # visualize_terrain_sub_meshes_3d(delatin_sub_meshes)

    for i, (vertices, faces) in enumerate(delatin_sub_meshes.values()):
        # Normalize and scale the X and Y coordinates of the vertices to fit the terrain
        vertices[:, :2] = (vertices[:, :2] / np.max(vertices[:, :2])) * WORLD_SIZE - WORLD_SIZE / 2

        exising_object_names = bpy.data.objects.keys()

        # Create a new mesh
        bpy_mesh = bpy.data.meshes.new(f"mesh")
        bpy_mesh.from_pydata(vertices=vertices, edges=[], faces=faces)
        bpy_mesh.update()  # Update the mesh to reflect the changes
        bpy_mesh.validate(verbose=True)  # Validate geometry and ensure it's correct

        # Create a new object and link it to the scene
        object_name = f"generated_mesh_{i}"
        mesh_object = bpy.data.objects.new(object_name, bpy_mesh)
        bpy.data.collections["Collection"].objects.link(mesh_object)
        bpy.context.view_layer.objects.active = mesh_object

        new_object_names = bpy.data.objects.keys()

        # Get the unique object names that were created
        unique_object_names = get_unique_object_names(
            existing_object_names=exising_object_names,
            new_object_names=new_object_names
        )

        for object_name in unique_object_names:
            apply_biome(
                bpy_object=bpy.data.objects[object_name],
                path=np.random.choice(grass_biomes),
            )

    save_as_blend_file(image_name=IMAGE_NAME)


if __name__ == "__main__":
    main()
