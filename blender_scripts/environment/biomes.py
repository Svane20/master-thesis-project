import bpy
from pathlib import Path
from typing import List, Union, Tuple, Set
import numpy as np

from constants.directories import BIOMES_DIRECTORY
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def get_all_biomes_by_directory(directory: Path = BIOMES_DIRECTORY) -> List[str]:
    """
    Get all biome files in the specified directory.

    Args:
        directory (Path): The directory to search for biome files.

    Returns:
        List[str]: A list of biome file paths.
    """
    paths = [str(f) for f in directory.rglob("*") if str(f).endswith(".biome")]

    logger.info(f"Found {len(paths)} biomes in {directory}")
    return paths


def apply_biomes_to_objects(
        unique_object_names: Set[str],
        biome_paths: List[str],
        density: Tuple[float, float] = (20.0, 30.0),
        label_index: int = 0,
) -> None:
    """
    Apply biomes to the specified objects.

    Args:
        unique_object_names (Set[str]): The set of unique object names.
        biome_paths (List[str]): The list of biome paths to apply.
        density (Tuple[float, float], optional): The density range for biome distribution.
        label_index (int, optional): The label index to assign to scattered objects.
    """
    logger.info(f"Applying biomes to {len(unique_object_names)} objects.")

    for object_name in unique_object_names:
        bpy_object: bpy.types.Object = bpy.data.objects.get(object_name)

        if bpy_object is None:
            logger.error(f"Object '{object_name}' not found in the scene.")
            continue

        random_biome_path = np.random.choice(biome_paths)
        logger.info(f"Applying biome from {random_biome_path} to object '{bpy_object.name}'.")

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
    Apply a specific biome to a Blender object.

    Args:
        bpy_object (Union[str, bpy.types.Object]): The Blender object or its name.
        biome_path (str): The path to the biome file to apply.
        density (Tuple[float, float], optional): The density range for biome scattering.
        label_index (int, optional): The pass index for the scattered objects.
    """
    # Ensure the correct object is referenced by name or direct object.
    if isinstance(bpy_object, str):
        bpy_object = bpy.data.objects.get(bpy_object)
        if bpy_object is None:
            logger.error(f"Object '{bpy_object}' not found in the scene.")
            raise ValueError(f"Object '{bpy_object}' not found in the scene.")

    logger.info(f"Applying biome '{biome_path}' to object '{bpy_object.name}'.")

    # Set the object as the new emitter for scattering
    bpy.ops.scatter5.set_new_emitter(obj_name=bpy_object.name)

    # Access the active scene (safer than hardcoding 'Scene')
    scene = bpy.context.scene

    # Apply random density within the given range
    scene.scatter5.operators.add_psy_density.f_distribution_density = np.random.uniform(*density)
    logger.info(f"Biome density set between {density[0]} and {density[1]}.")

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
        logger.info(f"Assigned pass index {label_index} to object '{scattered_object.name}'.")

    logger.info(f"Biome '{biome_path}' applied to object '{bpy_object.name}' successfully.")
