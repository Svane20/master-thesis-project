import bpy
from pathlib import Path
from typing import List, Union, Tuple, Set
import numpy as np
import logging

from utils.metadata import add_entry, MetadataKey


def get_all_biomes_by_directory(
        directory: str,
        include: List[str] | None = None,
        exclude: List[str] | None = None,
) -> List[str]:
    """
    Get all biome files in the specified directory.
    Args:
        directory (Path): The directory to search for biome files.
        include (List[str], optional): A list of keywords that must be present in the file path. Defaults to None.
        exclude (List[str], optional): A list of keywords to filter out from the file paths. Defaults to None.
    Returns:
        List[str]: A list of biome file paths.
    """
    paths = [str(f) for f in Path(directory).rglob("*") if str(f).endswith(".biome")]

    if include:
        paths = [path for path in paths if any(keyword in path for keyword in include)]

    if exclude:
        paths = [path for path in paths if not any(keyword in path for keyword in exclude)]

    logging.debug(f"Found {len(paths)} biomes in {directory}")
    return paths


def apply_biomes_to_objects(
        unique_object_names: Set[str],
        biome_paths: List[str],
        density: Tuple[float, float] = (20.0, 30.0),
        label_index: int = 0,
        seed: int = None,
) -> None:
    """
    Apply biomes to the specified objects.

    Args:
        unique_object_names (Set[str]): The set of unique object names.
        biome_paths (List[str]): The list of biome paths to apply.
        density (Tuple[float, float], optional): The density range for biome distribution. Defaults to (20.0, 30.0).
        label_index (int, optional): The label index to assign to scattered objects. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    logging.debug(f"Applying biomes to {len(unique_object_names)} objects.")

    if seed is not None:
        np.random.seed(seed)
        logging.debug(f"Seed set to {seed}")

    for object_name in unique_object_names:
        bpy_object = get_object_by_name(object_name)
        if bpy_object is None:
            continue

        random_biome_path = np.random.choice(biome_paths)
        add_entry(category=MetadataKey.BIOMES, filepath=Path(random_biome_path).as_posix())
        logging.info(f"Selected tree biome file: {random_biome_path}")

        apply_biome(
            bpy_object=bpy_object,
            biome_path=random_biome_path,
            density=density,
            label_index=label_index,
        )


def apply_biome(
        bpy_object: Union[str, bpy.types.Object],
        biome_path: str,
        density: Tuple[float, float] = (20.0, 30.0),
        label_index: int = 0,
) -> None:
    """
    Apply a biome to the given object.

    Args:
        bpy_object (Union[str, bpy.types.Object]): The Blender object or its name.
        biome_path (str): The path to the biome file to apply.
        density (Tuple[float, float], optional): The density range for biome scattering. Defaults to (20.0, 30.0).
        label_index (int, optional): The pass index for the scattered objects. Defaults to 0.

    Raises:
        ValueError: If the object is not found in the scene.
    """
    # Ensure the correct object is referenced by name or direct object.
    if isinstance(bpy_object, str):
        bpy_object = bpy.data.objects.get(bpy_object)
        if bpy_object is None:
            logging.error(f"Object '{bpy_object}' not found in the scene.")
            raise ValueError(f"Object '{bpy_object}' not found in the scene.")

    logging.debug(f"Applying biome '{biome_path}' to object '{bpy_object.name}'.")

    # Set the object as the new emitter for scattering
    bpy.ops.scatter5.set_new_emitter(obj_name=bpy_object.name)

    scene = bpy.context.scene

    # Apply random density within the given range
    scene.scatter5.operators.add_psy_density.f_distribution_density = np.random.uniform(*density)
    logging.debug(f"Biome density set between {density[0]} and {density[1]}.")

    # Force the view layer update
    bpy.context.view_layer.update()

    # Track objects before biome scattering
    before_scattering = set(bpy.data.objects.keys())

    # Add the biome to the emitter object
    bpy.ops.scatter5.add_biome(emitter_name=bpy_object.name, json_path=biome_path)

    # Track new objects created after scattering
    after_scattering = set(bpy.data.objects.keys())
    new_objects = after_scattering - before_scattering

    # Assign pass index to new scattered objects
    for object_name in new_objects:
        scattered_object: bpy.types.Object = bpy.data.objects[object_name]
        scattered_object.pass_index = label_index
        logging.debug(f"Assigned pass index {label_index} to object '{scattered_object.name}'.")

    logging.debug(f"Biome applied to object '{bpy_object.name}' successfully.")


def get_object_by_name(name: Union[str, bpy.types.Object]) -> Union[bpy.types.Object, None]:
    """
    Retrieve a Blender object by name.

    Args:
        name (Union[str, bpy.types.Object]): The object or its name.

    Returns:
        bpy.types.Object: The retrieved object or None if not found.
    """
    if isinstance(name, str):
        bpy_object = bpy.data.objects.get(name)
        if bpy_object is None:
            logging.error(f"Object '{name}' not found in the scene.")
            return None

        return bpy_object

    return name
