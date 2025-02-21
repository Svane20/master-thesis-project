import bpy
import numpy as np
from numpy.typing import NDArray
from pydelatin import Delatin
import trimesh
from typing import Tuple, Dict, List, Set
from pathlib import Path
import random
from uuid import uuid4
import os
import logging

from bpy_utils.bpy_ops import delete_object_by_selection
from environment.biomes import apply_biomes_to_objects, apply_biome, get_object_by_name


def convert_delatin_mesh_to_sub_meshes(
        mesh: Delatin,
        segmentation_map: NDArray[np.uint8]
) -> Dict[str, Tuple]:
    """
    Convert Delatin mesh into sub-meshes based on a segmentation map.

    Args:
        mesh (Delatin): Delatin object containing vertices and faces.
        segmentation_map (NDArray[np.uint8]): 3-channel segmentation map (RGB) indicating terrain categories.

    Returns:
        Dict[str, Tuple[NDArray[np.float32], NDArray[np.int32]]]: A dictionary mapping segmentation categories to sub-meshes.
    """
    logging.info("Converting Delatin mesh to sub-meshes...")

    vertices = np.copy(mesh.vertices)  # Copy vertices to avoid modifying original mesh
    faces = mesh.triangles
    logging.debug(f"Original mesh has {len(vertices)} vertices and {len(faces)} faces.")

    # Assign Z values to vertices based on segmentation categories
    vertices[:, 2] = _assign_vertex_z_values(vertices, segmentation_map)
    logging.debug("Assigned Z values to vertices based on segmentation categories.")

    # Classify faces into grass, texture, and beds based on Z values
    z_values_per_face = vertices[faces][:, :, 2]
    grass_faces, texture_faces, beds_faces = _classify_faces_by_z_value(faces, z_values_per_face)
    logging.info(
        f"Classified faces into categories: {len(grass_faces)} grass, {len(texture_faces)} texture, {len(beds_faces)} beds.")

    # Reset Z values in vertices to original
    vertices[:, 2] = mesh.vertices[:, 2]

    # Create sub-meshes for each category
    sub_meshes = {
        "grass": _create_trimesh_sub_mesh(vertices, grass_faces),
        "texture": _create_trimesh_sub_mesh(vertices, texture_faces),
        "beds": _create_trimesh_sub_mesh(vertices, beds_faces),
    }

    logging.info(f"Converted Delatin mesh to {len(sub_meshes)} sub-meshes successfully.")
    return sub_meshes


def generate_mesh_objects_from_delation_sub_meshes(
        world_size: int,
        delatin_sub_meshes: Dict[str, Tuple],
        tree_biomes_path: List[str],
        grass_biomes_path: List[str],
        not_grass_biomes_path: List[str],
        generate_trees: bool,
        tree_probability: float,
        grass_densities: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float]
        ] = ((60, 100.), (0., 0.1), (0.1, 5.)),
        tree_densities: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float]
        ] = ((0.001, 0.025), (0.01, 0.5), (0.01, 0.5)),
        biome_label_indices: Tuple[int, int, int] = (255, 0, 0),
        seed: int = None,
) -> None:
    """
    Generate mesh objects from Delatin sub-meshes, apply biomes, and delete the object after.
    Args:
        world_size (int, optional): The size of the world (terrain scaling).
        delatin_sub_meshes (Dict[str, Tuple]:): The Delatin sub-meshes (vertices and faces).
        tree_biomes_path (List[str]): The trees biomes path to be applied to the terrain sub-meshes.
        grass_biomes_path (List[str]): The grass biomes paths to be applied to the terrain sub-meshes.
        not_grass_biomes_path (List[str]): The not grass biomes paths to be applied to the terrain sub-meshes.
        generate_trees (bool, optional): Whether to generate a tree or not.
        tree_probability (float, optional): The probability of generating a tree.
        grass_densities (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], optional): The grass densities.
        tree_densities (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], optional): The tree densities.
        biome_label_indices (Tuple[int, int, int], optional): The biome label indices.
        seed (int, optional): The random seed.
    Raises:
        ValueError: If sub-meshes or biomes are not provided.
    """
    logging.info(f"Starting to generate {len(delatin_sub_meshes)} mesh objects from Delatin sub-meshes.")

    if not delatin_sub_meshes or not grass_biomes_path:
        raise ValueError("Sub-meshes or biome paths cannot be empty.")

    if seed is not None:
        np.random.seed(seed)
        logging.info(f"Seed set to {seed}")

    for i, (
            (vertices, faces),
            biomes_path_flag,
            density_grass,
            density_tree,
            biome_label_index
    ) in enumerate(
        zip(
            delatin_sub_meshes.values(),
            (grass_biomes_path, grass_biomes_path, not_grass_biomes_path),
            grass_densities,
            tree_densities,
            biome_label_indices
        )
    ):
        logging.debug(f"Processing sub-mesh {i} with {len(vertices)} vertices and {len(faces)} faces.")

        # Normalize and scale the X and Y coordinates of the vertices to fit the terrain
        vertices[:, :2] = (vertices[:, :2] / np.max(vertices[:, :2])) * world_size - world_size / 2
        # Get object names before creating new objects
        existing_object_names = bpy.data.objects.keys()
        # Create a new mesh
        bpy_mesh = bpy.data.meshes.new(f"generated_mesh_{i}")
        bpy_mesh.from_pydata(vertices=vertices, edges=[], faces=faces)
        bpy_mesh.update()
        bpy_mesh.validate(verbose=True)

        logging.info(f"Created mesh '{bpy_mesh.name}' for object '{i}'.")

        # Create a new object and link it to the scene
        object_name = f"generated_mesh_{i}"
        mesh_object = bpy.data.objects.new(object_name, bpy_mesh)
        bpy.data.collections["Collection"].objects.link(mesh_object)
        bpy.context.view_layer.objects.active = mesh_object

        logging.info(f"Mesh object '{object_name}' added to the scene.")

        # Get object names after creation
        new_object_names = bpy.data.objects.keys()
        unique_names = set(new_object_names) - set(existing_object_names)

        logging.debug(f"Unique object names created: {unique_names}")

        # Apply a random grass biome to the object
        if biomes_path_flag:
            logging.info(f"Applying grass biomes to {len(unique_names)} objects.")

            for o_name in set(new_object_names) - set(existing_object_names):
                bpy_object = get_object_by_name(o_name)
                if bpy_object is None:
                    continue

                random_biome_path = np.random.choice(biomes_path_flag)

                apply_biome(
                    bpy_object=bpy_object,
                    biome_path=random_biome_path,
                    density=density_grass,
                    label_index=biome_label_index,
                )

            logging.info(f"Applied grass biomes to object '{object_name}' with label index {biome_label_index}.")

        if generate_trees and random.random() < tree_probability:
            # Apply a random tree biome to the object
            apply_biomes_to_objects(
                unique_object_names=set(new_object_names) - set(existing_object_names),
                biome_paths=tree_biomes_path,
                density=density_tree,
                label_index=0,
            )
            logging.info(f"Applied tree biomes to object '{object_name}' with label index 0.")

        # Delete the object after applying the biome
        delete_object_by_selection(mesh_object)
        logging.info(f"Deleted object '{object_name}' after biome application.")


def populate_meshes(
        delatin_mesh: Delatin,
        delatin_sub_meshes: Dict[str, Tuple],
        texture_paths: List[str],
        world_size: float
) -> None:
    logging.info("Populating mesh objects.")

    vertices = delatin_mesh.vertices
    faces = delatin_mesh.triangles
    logging.debug(f"Mesh has {len(vertices)} vertices and {len(faces)} faces.")

    # Normalize and scale vertices to world size
    max_val = np.max(vertices[:, :2])
    vertices[:, :2] = vertices[:, :2] / max_val * world_size - world_size / 2
    logging.debug("Scaled vertices to fit world size.")

    # Create Blender mesh from delatin_mesh
    bpy_mesh = bpy.data.meshes.new("mesh")
    bpy_mesh.from_pydata(vertices, [], faces)
    bpy_mesh.update()
    bpy_mesh.validate()
    logging.info("Created Blender mesh from delatin_mesh.")

    # Create and link object
    obj = bpy.data.objects.new(f"label_this_object_{uuid4()}", bpy_mesh)
    bpy.data.collections["Collection"].objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    logging.info(f"Created and linked object {obj.name} to collection.")

    # Generate UV map for the mesh
    _generate_uv_map_for_planer_surface(
        bpy_mesh,
        vertices,
        faces,
        name=f"uv_map_{uuid4()}"
    )
    logging.info("Generated UV map for the initial mesh.")

    # Choose a random texture from available terrain textures
    logging.debug(f"Found {len(texture_paths)} textures")
    texture = str(random.choice(texture_paths))
    texture_name = Path(texture).name
    logging.info(f"Selected random texture: {texture_name}")

    # Determine texture resolution suffix and extract base name
    resolutions = ["_4k.blend", "_8k.blend", "_16k.blend"]
    for res in resolutions:
        if res in texture:
            name = texture_name.split(res)[0]
            logging.debug(f"Texture resolution '{res}' found. Using base name: {name}")
            break
    else:
        error_msg = f"Texture resolution not found in {texture}"
        logging.error(error_msg)
        assert False, error_msg

    # Append material from texture
    materials_before_appending = bpy.data.materials.keys()
    logging.info("Appending material from texture.")

    while True:
        try:
            bpy.ops.wm.append(
                filepath=texture,
                directory=os.path.join(texture, "Material"),
                filename=name,
            )
            materials_after_appending = bpy.data.materials.keys()
            new_materials = list(set(materials_after_appending) - set(materials_before_appending))
            logging.debug(f"Materials before appending: {materials_before_appending}")
            logging.debug(f"Materials after appending: {materials_after_appending}")
            logging.debug(f"New materials appended: {new_materials}")
            if not new_materials:
                raise Exception("No new materials appended.")
            name = new_materials
            break
        except Exception as e:
            logging.error(f"Failed to append {texture}. Error: {e}")
            break

    # Clear existing materials and add a new material slot
    obj.data.materials.clear()
    bpy.ops.object.material_slot_add()
    chosen_material = random.choice(name) if isinstance(name, list) else name
    bpy.data.materials[chosen_material].use_backface_culling = True
    bpy.data.materials[chosen_material].show_transparent_back = True
    obj.active_material = bpy.data.materials[chosen_material]
    logging.info(f"Assigned material '{chosen_material}' to object '{obj.name}'.")

    # Update mapping scale based on mesh dimensions
    min_val, max_val = _min_max_vertices(vertices)
    longest_dim = np.max(np.abs(max_val - min_val))
    longest_dim = max(longest_dim, 1)
    logging.debug(f"Mesh dimensions: min={min_val}, max={max_val}, longest_dim={longest_dim}")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.view_layer.objects.active = obj
    if "Mapping" in obj.active_material.node_tree.nodes:
        obj.active_material.node_tree.nodes["Mapping"].inputs["Scale"].default_value = (
            longest_dim,
            longest_dim,
            1,
        )
        logging.info("Updated material mapping scale in EDIT mode.")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Adjust submesh vertices for proper rendering (hack)
    logging.info("Adjusting submesh vertices for rendering hack.")
    submesh = list(delatin_sub_meshes.values())[0]
    submesh_vertices, submesh_faces = submesh
    submesh_vertices[:, 2] += 0.005
    logging.debug("Adjusted submesh vertices' z-coordinate by +0.005.")

    # Create new Blender mesh for the submesh
    bpy_mesh = bpy.data.meshes.new("mesh")
    bpy_mesh.from_pydata(submesh_vertices, [], submesh_faces)
    bpy_mesh.update()
    bpy_mesh.validate()
    logging.info("Created Blender mesh for submesh.")

    # Generate UV map for the submesh
    _generate_uv_map_for_planer_surface(
        bpy_mesh,
        submesh_vertices,
        submesh_faces,
        name=f"uv_map_{uuid4()}"
    )
    logging.info("Generated UV map for submesh.")

    # Create and link submesh object
    obj = bpy.data.objects.new(f"label_this_object_2_{uuid4()}", bpy_mesh)
    bpy.data.collections["Collection"].objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    logging.info(f"Created and linked submesh object {obj.name} to collection.")

    # Assign the same material to the submesh object
    obj.data.materials.clear()
    bpy.ops.object.material_slot_add()
    obj.active_material = bpy.data.materials[chosen_material]

    logging.info(f"Assigned material '{chosen_material}' to submesh object '{obj.name}'.")

    # Update mapping scale for the submesh
    min_val, max_val = _min_max_vertices(submesh_vertices)
    longest_dim = np.max(np.abs(max_val - min_val))
    longest_dim = max(longest_dim, 1)

    logging.debug(f"Submesh dimensions: min={min_val}, max={max_val}, longest_dim={longest_dim}")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.view_layer.objects.active = obj
    if "Mapping" in obj.active_material.node_tree.nodes:
        obj.active_material.node_tree.nodes["Mapping"].inputs["Scale"].default_value = (
            longest_dim,
            longest_dim,
            1,
        )
        logging.info("Updated submesh material mapping scale in EDIT mode.")

    bpy.ops.object.mode_set(mode="OBJECT")

    obj.pass_index = 255
    logging.info(f"Set pass index for object '{obj.name}' to 255.")


def _assign_vertex_z_values(
        vertices: NDArray[np.float32],
        segmentation_map: NDArray[np.uint8]
) -> NDArray[np.int64]:
    """
    Assign Z values to vertices based on segmentation map categories.

    Args:
        vertices (NDArray[np.float32]): Array of vertex positions.
        segmentation_map (NDArray[np.uint8]): Segmentation map with 3 categories (RGB).

    Returns:
        NDArray[np.int64]: Updated Z values for each vertex.
    """
    logging.debug("Assigning Z values to vertices based on segmentation map.")

    return np.select(
        [
            _is_category(vertices, segmentation_map, category_idx)
            for category_idx in range(3)  # 0: texture, 1: grass, 2: beds
        ],
        [1, 0, -1],  # Z values for texture, grass, and beds
        default=0
    )


def _is_category(
        vertices: NDArray[np.float32],
        segmentation_map: NDArray[np.uint8],
        category_idx: int
) -> NDArray[np.bool_]:
    """
    Check if each vertex belongs to a specific category in the segmentation map.

    Args:
        vertices (NDArray[np.float32]): Array of vertex positions.
        segmentation_map (NDArray[np.uint8]): Segmentation map.
        category_idx (int): Index of the category to check (0: texture, 1: grass, 2: beds).

    Returns:
        NDArray[np.bool_]: Boolean array indicating whether each vertex belongs to the given category.
    """
    x_coords = vertices[:, 0].astype(int)
    y_coords = vertices[:, 1].astype(int)
    logging.debug(f"Checking category {category_idx} for vertices.")

    return np.asarray(segmentation_map[y_coords, x_coords, category_idx] == 255)


def _classify_faces_by_z_value(
        faces: NDArray[np.uint32],
        z_values_per_face: NDArray[np.float32]
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Classify faces based on the Z values of their vertices.

    Args:
        faces (NDArray[np.uint32]): Array of mesh faces.
        z_values_per_face (NDArray[np.float32]): Z values for each face's vertices.

    Returns:
        Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]: A tuple of arrays for grass faces, texture faces, and beds faces.
    """
    logging.debug("Classifying faces by Z values.")
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

    logging.debug(
        f"Classified {len(grass_faces)} grass faces, {len(texture_faces)} texture faces, and {len(beds_faces)} beds faces."
    )

    return np.array(grass_faces), np.array(texture_faces), np.array(beds_faces)


def _create_trimesh_sub_mesh(
        vertices: NDArray[np.float32],
        faces: NDArray[np.uint32]
) -> Tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """
    Creates a trimesh-compatible sub-mesh.

    Args:
        vertices (NDArray[np.float32]): Array of mesh vertices.
        faces (NDArray[np.uint32]): Faces for the sub-mesh.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.uint64]]: A tuple of (vertices, faces) arrays representing the sub-mesh.
    """
    if len(faces) == 0:
        logging.debug("No faces found for this sub-mesh.")
        return np.array([]), np.array([])  # Return empty arrays if no faces exist

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    logging.info(f"Created trimesh sub-mesh with {len(faces)} faces.")

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


def _generate_uv_map_for_planer_surface(
        bpy_mesh: bpy.types.Mesh,
        vertices: list[tuple],
        faces: list[tuple],
        name: str = "UVMap",
) -> None:
    logging.info(f"Generating UV map '{name}' for mesh.")

    min_val, max_val = _min_max_vertices(vertices)

    logging.debug(f"UV mapping: min vertices {min_val}, max vertices {max_val}")

    x_dim = max_val[0] - min_val[0]
    y_dim = max_val[1] - min_val[1]
    max_dim = max(x_dim, y_dim)

    logging.debug(f"UV mapping dimensions: x_dim={x_dim}, y_dim={y_dim}, max_dim={max_dim}")

    uvs = []
    for ff in faces:
        for f in ff:
            vertex = vertices[f]
            uvs.extend([(vertex[0] - min_val[0]) / max_dim, (vertex[1] - min_val[1]) / max_dim])

    uv_layer = bpy_mesh.uv_layers.new(name=name)
    uv_layer.data.foreach_set("uv", uvs)

    logging.info(f"UV map '{name}' created with {len(uvs) // 2} coordinate pairs.")


def _min_max_vertices(vertices: list[tuple]) -> tuple:
    vertices = np.array(vertices)
    min_val = np.min(vertices, axis=0)
    max_val = np.max(vertices, axis=0)

    logging.debug(f"Calculated min_val: {min_val}, max_val: {max_val} for vertices.")

    return min_val, max_val
