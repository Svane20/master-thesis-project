import bpy

from pathlib import Path
from typing import List, Union, Tuple, Set
import numpy as np

from constants.directories import BIOMES_DIRECTORY
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def get_all_biomes_by_directory(directory: Path = BIOMES_DIRECTORY) -> List[str]:
    """
    Get all biomes in the specified directory.

    Args:
        directory: The directory to search for biomes.

    Returns:
        A list of biome paths.
    """
    paths = [str(f) for f in directory.rglob("*") if str(f).endswith(".biome")]

    logger.info(f"Found {len(paths)} biomes in {directory}")

    return paths


def apply_biomes_to_objects(
        unique_object_names: Set[str],
        biome_paths: List[str],
        density: Tuple[float, float] = (20., 30.),
        label_index: int = 0,
) -> None:
    """
    Apply biomes to the newly created objects.

    Args:
        unique_object_names: The unique object names.
        biome_paths: The biomes paths.
        density: The density of the biome.
        label_index: The label index.
    """
    for object_name in unique_object_names:
        bpy_object: bpy.types.Object = bpy.data.objects[object_name]

        # Apply a random biome to the object
        random_biome_path = np.random.choice(biome_paths)

        apply_biome(
            bpy_object=bpy_object,
            biome_path=random_biome_path,
            density=density,
            label_index=label_index,
        )


def apply_biome(
        bpy_object: Union[str, bpy.types.Object],
        biome_path: str,
        density: Tuple[float, float] = (20., 30.),
        label_index: int = 0,
) -> None:
    """
    Apply a biome to a Blender object.

    Args:
        bpy_object: The Blender object to apply the biome to.
        biome_path: The path to the biome.
        density: The density of the biome.
        label_index: The label index.
    """
    # Ensure the correct object is referenced
    if isinstance(bpy_object, str):
        bpy_object = bpy.data.objects.get(bpy_object)
        if bpy_object is None:
            raise ValueError(f"Object '{bpy_object}' not found in the scene.")

    logger.info(f"Applying biome {biome_path} to {bpy_object.name}")

    # Set the object as the new emitter
    bpy.ops.scatter5.set_new_emitter(obj_name=bpy_object.name)

    # Access the active scene (safer than hardcoding 'Scene')
    scene = bpy.context.scene

    # Apply random density distribution
    scene.scatter5.operators.add_psy_density.f_distribution_density = np.random.uniform(*density)

    # Force view layer update
    bpy.context.view_layer.update()

    # Track new objects added after scattering
    before_scattering = set(bpy.data.objects.keys())

    # Add the biome to the emitter
    bpy.ops.scatter5.add_biome(emitter_name=bpy_object.name, json_path=biome_path)

    after_scattering = set(bpy.data.objects.keys())
    new_objects = after_scattering - before_scattering

    # Assign pass index to new scattered objects
    for object_name in new_objects:
        scattered_object: bpy.types.Object = bpy.data.objects[object_name]
        scattered_object.pass_index = label_index

    logger.info(f"Biome {biome_path} applied to {bpy_object.name} successfully.")
