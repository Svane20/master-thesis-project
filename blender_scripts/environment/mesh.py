import bpy

import numpy as np
from pydelatin import Delatin
import trimesh
from typing import Tuple, Dict, List, Set

from bpy_utils.bpy_ops import delete_object_by_selection
from constants.defaults import WorldDefaults
from custom_logging.custom_logger import setup_logger
from environment.biomes import apply_biomes_to_objects

logger = setup_logger(__name__)

DensityRange = Tuple[float, float]  # Alias for readability


def convert_delatin_mesh_to_sub_meshes(
        mesh: Delatin,
        segmentation_map: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert Delatin mesh into sub-meshes based on a segmentation map.

    Args:
        mesh (Delatin): Delatin object containing vertices and faces.
        segmentation_map (np.ndarray): 3-channel segmentation map (RGB) indicating terrain categories.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: A dictionary mapping segmentation categories to sub-meshes.
    """
    logger.info("Converting Delatin mesh to sub-meshes...")

    vertices = np.copy(mesh.vertices)  # Copy vertices to avoid modifying original mesh
    faces = mesh.triangles
    logger.debug(f"Original mesh has {len(vertices)} vertices and {len(faces)} faces.")

    # Assign Z values to vertices based on segmentation categories
    vertices[:, 2] = _assign_vertex_z_values(vertices, segmentation_map)
    logger.debug("Assigned Z values to vertices based on segmentation categories.")

    # Classify faces into grass, texture, and beds based on Z values
    z_values_per_face = vertices[faces][:, :, 2]
    grass_faces, texture_faces, beds_faces = _classify_faces_by_z_value(faces, z_values_per_face)
    logger.info(
        f"Classified faces into categories: {len(grass_faces)} grass, {len(texture_faces)} texture, {len(beds_faces)} beds.")

    # Reset Z values in vertices to original
    vertices[:, 2] = mesh.vertices[:, 2]

    # Create sub-meshes for each category
    sub_meshes = {
        "grass": _create_trimesh_sub_mesh(vertices, grass_faces),
        "texture": _create_trimesh_sub_mesh(vertices, texture_faces),
        "beds": _create_trimesh_sub_mesh(vertices, beds_faces),
    }

    logger.info(f"Converted Delatin mesh to {len(sub_meshes)} sub-meshes successfully.")
    return sub_meshes


def generate_mesh_objects_from_delation_sub_meshes(
        delatin_sub_meshes: Dict[str, Tuple[np.ndarray, np.ndarray]],
        biomes_paths: List[str],
        grass_densities: Tuple[DensityRange, DensityRange, DensityRange] = ((60, 100.), (0., 0.1), (0.1, 5.)),
        biome_label_indices: Tuple[int, int, int] = (255, 0, 0),
        world_size: int = WorldDefaults.SIZE,
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


def _assign_vertex_z_values(vertices: np.ndarray, segmentation_map: np.ndarray) -> np.ndarray:
    """
    Assign Z values to vertices based on segmentation map categories.

    Args:
        vertices (np.ndarray): Array of vertex positions.
        segmentation_map (np.ndarray): Segmentation map with 3 categories (RGB).

    Returns:
        np.ndarray: Updated Z values for each vertex.
    """
    logger.debug("Assigning Z values to vertices based on segmentation map.")
    return np.select(
        [
            _is_category(vertices, segmentation_map, category_idx)
            for category_idx in range(3)  # 0: texture, 1: grass, 2: beds
        ],
        [1, 0, -1],  # Z values for texture, grass, and beds
        default=0
    )


def _is_category(vertices: np.ndarray, segmentation_map: np.ndarray, category_idx: int) -> np.ndarray:
    """
    Check if each vertex belongs to a specific category in the segmentation map.

    Args:
        vertices (np.ndarray): Array of vertex positions.
        segmentation_map (np.ndarray): Segmentation map.
        category_idx (int): Index of the category to check (0: texture, 1: grass, 2: beds).

    Returns:
        np.ndarray: Boolean array indicating whether each vertex belongs to the given category.
    """
    x_coords = vertices[:, 0].astype(int)
    y_coords = vertices[:, 1].astype(int)
    logger.debug(f"Checking category {category_idx} for vertices.")
    return np.asarray(segmentation_map[x_coords, y_coords, category_idx] == 255)


def _classify_faces_by_z_value(faces: np.ndarray, z_values_per_face: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify faces based on the Z values of their vertices.

    Args:
        faces (np.ndarray): Array of mesh faces.
        z_values_per_face (np.ndarray): Z values for each face's vertices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of arrays for grass faces, texture faces, and beds faces.
    """
    logger.debug("Classifying faces by Z values.")
    grass_faces = []
    texture_faces = []
    beds_faces = []

    for i, z_values in enumerate(z_values_per_face):
        if np.any(z_values == 0):  # grass
            grass_faces.append(faces[i])
        elif np.any(z_values == 1):  # texture
            texture_faces.append(faces[i])
        elif np.any(z_values == -1):  # beds
            beds_faces.append(faces[i])

    logger.debug(
        f"Classified {len(grass_faces)} grass faces, {len(texture_faces)} texture faces, and {len(beds_faces)} beds faces."
    )
    return np.array(grass_faces), np.array(texture_faces), np.array(beds_faces)


def _create_trimesh_sub_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a trimesh-compatible sub-mesh.

    Args:
        vertices (np.ndarray): Array of mesh vertices.
        faces (np.ndarray): Faces for the sub-mesh.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of (vertices, faces) arrays representing the sub-mesh.
    """
    if len(faces) == 0:
        logger.debug("No faces found for this sub-mesh.")
        return np.array([]), np.array([])  # Return empty arrays if no faces exist

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    logger.info(f"Created trimesh sub-mesh with {len(faces)} faces.")
    return np.array(mesh.vertices), np.array(mesh.faces)


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
