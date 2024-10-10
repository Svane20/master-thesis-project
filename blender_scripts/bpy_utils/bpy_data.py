import bpy

from pathlib import Path
from typing import Dict
from enum import Enum

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

SCENE = "Scene"
VIEW_LAYER = "ViewLayer"


class BlendFilePropertyKey(Enum):
    Collections = "collections"
    Objects = "objects"
    Meshes = "meshes"


def list_data_blocks_in_blend_file(blend_file: Path, key: BlendFilePropertyKey) -> Dict[str, list]:
    """
    Lists all data blocks (e.g., collections, objects) inside a .blend file based on the provided key.

    Args:
        blend_file: Path to the .blend file.
        key: The specific data block key to list (e.g., 'collections', 'objects', 'meshes').

    Returns:
        A dictionary with data block names as keys and an empty list as values.
    """
    data_blocks_dict = {}

    logger.info(f"Loading '{key.value}' from blend file: {blend_file}")

    try:
        with bpy.data.libraries.load(str(blend_file), link=False) as (data_from, data_to):
            # Use the Enum's value for dynamic access
            if hasattr(data_from, key.value):
                data_block = getattr(data_from, key.value)

                for item in data_block:
                    data_blocks_dict[item] = []

                logger.info(f"Loaded {key.value}: {list(data_blocks_dict.keys())} from blend file: {blend_file}")
            else:
                logger.error(f"The key '{key.value}' does not exist in the blend file.")

        return data_blocks_dict
    except Exception as e:
        logger.error(f"Error loading blend file: {e}")
        raise


def set_scene_alpha_threshold(alpha_threshold: float = 0.5) -> None:
    """
    Set the alpha threshold for the scene.

    Args:
        alpha_threshold: The alpha threshold.
    """
    bpy.data.scenes[SCENE].view_layers[VIEW_LAYER].pass_alpha_threshold = alpha_threshold


def use_backface_culling_on_materials(use_backface_culling: bool = True) -> None:
    """
    Set backface culling on all materials.

    Args:
        use_backface_culling: Whether to enable backface culling.
    """

    for material in bpy.data.materials:
        material.use_backface_culling = use_backface_culling
        logger.info(f"Set backface culling to {use_backface_culling} for material: {material.name}")
