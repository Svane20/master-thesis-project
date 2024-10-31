import bpy

from pathlib import Path
from typing import Dict
from enum import Enum

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

SCENE = "Scene"
VIEW_LAYER = "ViewLayer"


class BlendFilePropertyKey(str, Enum):
    """
    Enum for selecting specific data blocks from a blend file.

    Attributes:
        Collections: Access collections from the blend file.
        Objects: Access objects from the blend file.
        Meshes: Access meshes from the blend file.
    """
    Collections = "collections"
    Objects = "objects"
    Meshes = "meshes"


def list_data_blocks_in_blend_file(blend_file: Path, key: BlendFilePropertyKey) -> Dict[str, list]:
    """
    Lists all data blocks (e.g., collections, objects) inside a .blend file based on the provided key.

    Args:
        blend_file (Path): Path to the .blend file.
        key (BlendFilePropertyKey): The specific data block key to list (e.g., 'collections', 'objects', 'meshes').

    Returns:
        Dict[str, list]: A dictionary with data block names as keys and an empty list as values.

    Raises:
        Exception: If there’s an error loading the blend file or if the key is not found.
    """
    data_blocks_dict = {}
    logger.info(f"Attempting to load '{key.value}' from blend file: {blend_file}")

    try:
        with bpy.data.libraries.load(str(blend_file), link=False) as (data_from, data_to):
            if hasattr(data_from, key.value):
                data_block = getattr(data_from, key.value)
                data_blocks_dict = {item: [] for item in data_block}
                logger.info(f"Loaded {key.value}: {list(data_blocks_dict.keys())} from blend_file: {blend_file}")
            else:
                logger.warning(f"Blend file does not contain '{key.value}': {blend_file}")
    except Exception as e:
        logger.error(f"Failed to load blend file: {blend_file}. Error: {e}")
        raise

    return data_blocks_dict


def set_scene_alpha_threshold(
        alpha_threshold: float = 0.5,
        scene_name: str = SCENE,
        view_layer_name: str = VIEW_LAYER
) -> None:
    """
    Sets the alpha threshold for the specified scene's view layer.

    Args:
        alpha_threshold (float): The alpha threshold value to set. Defaults to 0.5.
        scene_name (str): The name of the scene. Defaults to "Scene".
        view_layer_name (str): The name of the view layer. Defaults to "ViewLayer".

    Raises:
        KeyError: If the scene or view layer does not exist.
    """
    try:
        bpy.data.scenes[scene_name].view_layers[view_layer_name].pass_alpha_threshold = alpha_threshold
        logger.info(
            f"Set alpha threshold to {alpha_threshold} for scene '{scene_name}' and view layer '{view_layer_name}'.")
    except KeyError as e:
        logger.error(f"Failed to set alpha threshold: {e}")
        raise


def use_backface_culling_on_materials(use_backface_culling: bool = True) -> None:
    """
    Enables or disables backface culling on all materials in the current Blender project.

    Args:
        use_backface_culling (bool): Whether to enable or disable backface culling. Defaults to True.

    Logs:
        Logs each material's name and the updated backface culling status.
    """
    for material in bpy.data.materials:
        material.use_backface_culling = use_backface_culling
        logger.debug(f"Set backface culling to {use_backface_culling} for material: {material.name}")
