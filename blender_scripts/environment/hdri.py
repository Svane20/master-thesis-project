import bpy
import numpy as np
from pathlib import Path
from typing import List
import random

from constants.directories import HDRI_PURE_SKIES_DIRECTORY
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

# Constants
HDRI_EXTENSIONS = [".exr", ".hdr"]
MIN_TEMPERATURE = 5000
MAX_TEMPERATURE = 6500
MIN_STRENGTH = 0.6
MAX_STRENGTH = 1.0
MIN_SUN_SIZE = 1
MAX_SUN_SIZE = 3
MIN_SUN_ELEVATION = 45
MAX_SUN_ELEVATION = 90
MIN_SUN_ROTATION = 0
MAX_SUN_ROTATION = 360
MIN_SUN_INTENSITY = 0.4
MAX_SUN_INTENSITY = 0.8
MIN_DENSITY = 0
MAX_DENSITY = 2


def get_all_hdri_by_directory(directory: Path = HDRI_PURE_SKIES_DIRECTORY) -> List[Path]:
    """
    Retrieve all HDRI files in the specified directory.

    Args:
        directory (Path): The directory to search for HDRI files.

    Returns:
        List[Path]: A list of HDRI file paths.

    Raises:
        FileNotFoundError: If no HDRI files are found in the directory.
    """
    hdri_files = []
    for ext in HDRI_EXTENSIONS:
        hdri_files += list(directory.glob(f"*{ext}"))

    if not hdri_files:
        logger.error(f"No HDRI files found in {directory}")
        raise FileNotFoundError(f"No HDRI files found in {directory}")

    logger.info(f"Found {len(hdri_files)} HDRI files in {directory}")
    return hdri_files


def add_sky_to_scene(directory: Path = HDRI_PURE_SKIES_DIRECTORY) -> None:
    """
    Add a random sky (HDRI or sky texture) to the Blender scene.

    Args:
        directory (Path): The directory containing HDRI files.
    """
    logger.info(f"Adding sky to the scene from directory: {directory}")
    hdri_paths = get_all_hdri_by_directory(directory)
    random_hdri_path = random.choice(hdri_paths)
    logger.info(f"Selected HDRI file: {random_hdri_path}")

    node_tree = bpy.context.scene.world.node_tree
    tree_nodes = node_tree.nodes
    logger.debug("Clearing existing nodes from the world node tree.")
    tree_nodes.clear()

    logger.debug("Adding HDRI to the scene.")
    add_hdri(random_hdri_path, tree_nodes, node_tree)

    logger.debug("Adding procedural sky texture to the scene.")
    add_sky_texture(tree_nodes, node_tree)

    logger.debug("Setting up world output shader.")
    setup_world_output(tree_nodes, node_tree)


def add_hdri(path: Path, tree_nodes, node_tree) -> None:
    """
    Add HDRI node setup to the node tree.

    Args:
        path (Path): The path to the HDRI file.
        tree_nodes: The tree nodes of the scene.
        node_tree: The node tree to which the HDRI nodes will be added.
    """
    logger.info(f"Adding HDRI node for {path}")

    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_background.inputs["Strength"].default_value = clamp(random.uniform(MIN_STRENGTH, MAX_STRENGTH), MIN_STRENGTH,
                                                             MAX_STRENGTH)
    logger.debug(f"Set background strength to {node_background.inputs['Strength'].default_value}")

    node_environment = tree_nodes.new("ShaderNodeTexEnvironment")
    node_environment.image = bpy.data.images.load(path.as_posix())
    logger.debug("Loaded HDRI image.")

    node_blackbody = tree_nodes.new("ShaderNodeBlackbody")
    node_blackbody.inputs["Temperature"].default_value = random.randint(MIN_TEMPERATURE, MAX_TEMPERATURE)
    logger.debug(f"Set blackbody temperature to {node_blackbody.inputs['Temperature'].default_value}")

    node_multiply = tree_nodes.new("ShaderNodeVectorMath")
    node_multiply.operation = "MULTIPLY"
    logger.debug("Added multiply operation for vector math.")

    links = node_tree.links
    links.new(node_multiply.outputs["Vector"], node_background.inputs["Color"])
    links.new(node_blackbody.outputs["Color"], node_multiply.inputs[0])
    links.new(node_environment.outputs["Color"], node_multiply.inputs[1])
    logger.info("Linked HDRI nodes successfully.")


def add_sky_texture(tree_nodes, node_tree) -> None:
    """
    Add a procedural sky texture node setup to the node tree.

    Args:
        tree_nodes: The tree nodes of the scene.
        node_tree: The node tree to which the sky texture nodes will be added.
    """
    logger.info("Adding procedural sky texture.")

    node_sky = tree_nodes.new(type="ShaderNodeTexSky")
    node_sky.sky_type = "NISHITA"
    node_sky.sun_size = np.deg2rad(random.uniform(MIN_SUN_SIZE, MAX_SUN_SIZE))
    node_sky.sun_elevation = np.deg2rad(random.uniform(MIN_SUN_ELEVATION, MAX_SUN_ELEVATION))
    node_sky.sun_rotation = np.deg2rad(random.uniform(MIN_SUN_ROTATION, MAX_SUN_ROTATION))
    node_sky.sun_intensity = random.uniform(MIN_SUN_INTENSITY, MAX_SUN_INTENSITY)
    logger.debug(f"Sky texture parameters - Sun size: {node_sky.sun_size}, Elevation: {node_sky.sun_elevation}, "
                 f"Rotation: {node_sky.sun_rotation}, Intensity: {node_sky.sun_intensity}")

    node_sky.altitude = random.randint(MIN_DENSITY, MAX_DENSITY)
    node_sky.air_density = clamp(random.randint(MIN_DENSITY, MAX_DENSITY), MIN_DENSITY, MAX_DENSITY)
    node_sky.dust_density = clamp(random.randint(MIN_DENSITY, MAX_DENSITY), MIN_DENSITY, MAX_DENSITY)
    node_sky.ozone_density = clamp(random.randint(MIN_DENSITY, MAX_DENSITY), MIN_DENSITY, MAX_DENSITY)
    logger.debug(
        f"Sky texture atmospheric values - Altitude: {node_sky.altitude}, Air density: {node_sky.air_density}, "
        f"Dust density: {node_sky.dust_density}, Ozone density: {node_sky.ozone_density}")

    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_background.inputs["Strength"].default_value = clamp(random.uniform(0.2, 0.6), 0.0, 1.0)
    logger.debug(f"Set background strength to {node_background.inputs['Strength'].default_value}")

    links = node_tree.links
    links.new(node_sky.outputs["Color"], node_background.inputs["Color"])
    logger.info("Linked sky texture nodes successfully.")


def setup_world_output(tree_nodes, node_tree) -> None:
    """
    Set up the output shader for the world background in the node tree.

    Args:
        tree_nodes: The tree nodes of the scene.
        node_tree: The node tree to which the output shader will be added.
    """
    logger.info("Setting up world output shader.")

    node_add = tree_nodes.new(type="ShaderNodeAddShader")
    links = node_tree.links
    k = 0

    for node in tree_nodes:
        if "Background" in node.name:
            links.new(node.outputs["Background"], node_add.inputs[k])
            k += 1

    node_output = tree_nodes.new(type="ShaderNodeOutputWorld")
    links.new(node_add.outputs["Shader"], node_output.inputs["Surface"])
    logger.info("World output shader set up successfully.")


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between a minimum and maximum value.

    Args:
        value (float): The value to clamp.
        min_value (float): The minimum allowed value.
        max_value (float): The maximum allowed value.

    Returns:
        float: The clamped value.
    """
    return max(min(value, max_value), min_value)
