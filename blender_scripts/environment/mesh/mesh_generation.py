import bpy
import numpy as np
from typing import Dict, List, Tuple, Set
from environment.biomes import apply_biomes_to_objects
from constants.defaults import WORLD_SIZE
from bpy_utils.bpy_ops import delete_object_by_selection
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

DensityRange = Tuple[float, float]  # Alias for readability


def generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes: Dict[str, Tuple[np.ndarray, np.ndarray]],
        biomes_paths: List[str],
        grass_densities: Tuple[DensityRange, DensityRange, DensityRange] = ((60, 100.), (0., 0.1), (0.1, 5.)),
        biome_label_indices: Tuple[int, int, int] = (255, 0, 0),
        world_size: int = WORLD_SIZE,
) -> None:
    """
    Generate mesh objects from Delatin sub-meshes, apply biomes, and delete the object after.

    Args:
        delatin_sub_meshes (Dict[str, Tuple[np.ndarray, np.ndarray]]): The Delatin sub-meshes (vertices and faces).
        biomes_paths (List[str]): The biomes paths to be applied to the objects.
        grass_densities (Tuple[DensityRange, DensityRange, DensityRange], optional): The grass densities.
        biome_label_indices (Tuple[int, int, int], optional): The biome label indices.
        world_size (int, optional): The size of the world (terrain scaling).

    Raises:
        ValueError: If sub-meshes or biomes are not provided.
    """
    logger.info(f"Starting to generate {len(delatin_sub_meshes)} mesh objects from Delatin sub-meshes.")

    if not delatin_sub_meshes or not biomes_paths:
        raise ValueError("Sub-meshes or biome paths cannot be empty.")

    for i, ((vertices, faces), density_grass, biome_label_index) in enumerate(
            zip(delatin_sub_meshes.values(), grass_densities, biome_label_indices)
    ):
        logger.debug(f"Processing sub-mesh {i} with {len(vertices)} vertices and {len(faces)} faces.")

        # Normalize and scale the X and Y coordinates of the vertices to fit the terrain
        vertices[:, :2] = (vertices[:, :2] / np.max(vertices[:, :2])) * world_size - world_size / 2

        # Get object names before creating new objects
        existing_object_names = bpy.data.objects.keys()

        # Create a new mesh
        bpy_mesh = bpy.data.meshes.new(f"generated_mesh_{i}")
        bpy_mesh.from_pydata(vertices=vertices, edges=[], faces=faces)
        bpy_mesh.update()
        bpy_mesh.validate(verbose=True)
        logger.info(f"Created mesh '{bpy_mesh.name}' for object '{i}'.")

        # Create a new object and link it to the scene
        object_name = f"generated_mesh_{i}"
        mesh_object = bpy.data.objects.new(object_name, bpy_mesh)
        bpy.data.collections["Collection"].objects.link(mesh_object)
        bpy.context.view_layer.objects.active = mesh_object
        logger.info(f"Mesh object '{object_name}' added to the scene.")

        # Get object names after creation
        new_object_names = bpy.data.objects.keys()

        # Get the unique object names that were created
        unique_object_names = _get_unique_object_names(
            existing_object_names=existing_object_names,
            new_object_names=new_object_names
        )
        logger.debug(f"Unique object names created: {unique_object_names}")

        # Apply a random biome to the object
        apply_biomes_to_objects(
            unique_object_names=unique_object_names,
            biome_paths=biomes_paths,
            density=density_grass,
            label_index=biome_label_index,
        )
        logger.info(f"Applied biomes to object '{object_name}' with label index {biome_label_index}.")

        # Delete the object after applying the biome
        delete_object_by_selection(mesh_object)
        logger.info(f"Deleted object '{object_name}' after biome application.")


def _get_unique_object_names(existing_object_names: List[str], new_object_names: List[str]) -> Set[str]:
    """
    Get the unique object names that were created.

    Args:
        existing_object_names (List[str]): The existing object names before new objects were created.
        new_object_names (List[str]): The new object names after new objects were created.

    Returns:
        Set[str]: The set of unique object names that were created.
    """
    return set(new_object_names) - set(existing_object_names)
