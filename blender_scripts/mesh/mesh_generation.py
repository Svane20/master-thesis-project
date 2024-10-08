import bpy

import numpy as np

from typing import Dict, List, Tuple, Set

from biomes.biomes import apply_biomes_to_objects
from configuration.consts import Constants
from engine.bpy_ops import delete_object_by_selection

DensityRange = Tuple[float, float]  # Alias for readability


def generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes: Dict[str, Tuple[np.ndarray, np.ndarray]],
        biomes_paths: List[str],
        grass_densities: Tuple[DensityRange, DensityRange, DensityRange] = ((60, 100.), (0., 0.1), (0.1, 5.)),
        biome_label_indices: Tuple[int, int, int] = (255, 0, 0),
        world_size: int = Constants.Default.WORLD_SIZE,
) -> None:
    """
    Generate mesh objects from Delatin sub meshes.

    Args:
        delatin_sub_meshes: The Delatin sub meshes.
        biomes_paths: The biomes paths.
        grass_densities: The grass densities.
        biome_label_indices: The biome label indices.
        world_size: The world size.
    """
    for i, (
            (vertices, faces),
            density_grass,
            biome_label_index
    ) in enumerate(
        zip(
            delatin_sub_meshes.values(),  # Mesh vertices and faces
            grass_densities,
            biome_label_indices
        )
    ):
        # Normalize and scale the X and Y coordinates of the vertices to fit the terrain
        vertices[:, :2] = (vertices[:, :2] / np.max(vertices[:, :2])) * world_size - world_size / 2

        # Before creating object
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

        # After creating object
        new_object_names = bpy.data.objects.keys()

        # Get the unique object names that were created
        unique_object_names = _get_unique_object_names(
            existing_object_names=exising_object_names,
            new_object_names=new_object_names
        )

        # Apply a random biome to the object
        apply_biomes_to_objects(
            unique_object_names=unique_object_names,
            biome_paths=biomes_paths,
            density=density_grass,
            label_index=biome_label_index,
        )

        # Delete the object to place a new one
        delete_object_by_selection(mesh_object)


def _get_unique_object_names(existing_object_names: List[str], new_object_names: List[str]) -> Set[str]:
    """
    Get the unique object names that were created.

    Args:
        existing_object_names: The existing object names.
        new_object_names: The new object names.

    Returns:
        The unique object names that were created.
    """
    return set(new_object_names) - set(existing_object_names)
