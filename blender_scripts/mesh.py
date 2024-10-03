import numpy as np
from pydelatin import Delatin
import trimesh

from typing import Tuple, Dict


def convert_delatin_mesh_to_sub_meshes(mesh: Delatin, segmentation_map: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert Delatin mesh into sub-meshes based on a segmentation map.

    Args:
        mesh: Delatin object containing vertices and faces.
        segmentation_map: 3-channel segmentation map (RGB) indicating terrain categories.

    Returns:
        A dictionary mapping segmentation categories to sub-meshes.
    """
    vertices = np.copy(mesh.vertices)  # Copy vertices to avoid modifying original mesh
    faces = mesh.triangles

    # Assign Z values to vertices based on segmentation categories (1: texture, 0: grass, -1: beds)
    vertices[:, 2] = _assign_vertex_z_values(vertices, segmentation_map)

    # Classify faces into grass, not grass (texture), and beds based on Z values
    z_values_per_face = vertices[faces][:, :, 2]
    grass_faces, texture_faces, beds_faces = _classify_faces_by_z_value(faces, z_values_per_face)

    # Reset Z values in vertices to original
    vertices[:, 2] = mesh.vertices[:, 2]

    # Create sub-meshes for each category
    sub_meshes = {
        "grass": _create_trimesh_sub_mesh(vertices, grass_faces),
        "texture": _create_trimesh_sub_mesh(vertices, texture_faces),
        "beds": _create_trimesh_sub_mesh(vertices, beds_faces),
    }

    return sub_meshes


def _assign_vertex_z_values(vertices: np.ndarray, segmentation_map: np.ndarray) -> np.ndarray:
    """
    Assign Z values to vertices based on segmentation map categories.

    Args:
        vertices: Array of vertex positions.
        segmentation_map: Segmentation map with 3 categories (RGB).

    Returns:
        Updated Z values for each vertex.
    """
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
        vertices: Array of vertex positions.
        segmentation_map: Segmentation map.
        category_idx: Index of the category to check (0: texture, 1: grass, 2: beds).

    Returns:
        A boolean array indicating whether each vertex belongs to the given category.
    """
    x_coords = vertices[:, 0].astype(int)
    y_coords = vertices[:, 1].astype(int)
    return np.asarray(segmentation_map[x_coords, y_coords, category_idx] == 255)


def _classify_faces_by_z_value(faces: np.ndarray, z_values_per_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify faces based on the Z values of their vertices.

    Args:
        faces: Array of mesh faces.
        z_values_per_face: Z values for each face's vertices.

    Returns:
        A tuple of lists: (grass faces, texture faces, beds faces).
    """
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

    return np.array(grass_faces), np.array(texture_faces), np.array(beds_faces)


def _create_trimesh_sub_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a trimesh-compatible submesh.

    Args:
        vertices: Array of mesh vertices.
        faces: Faces for the submesh.

    Returns:
        A tuple of (vertices, faces) arrays representing the submesh.
    """
    if len(faces) == 0:
        return np.array([]), np.array([])  # Return empty arrays if no faces exist

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return np.array(mesh.vertices), np.array(mesh.faces)
