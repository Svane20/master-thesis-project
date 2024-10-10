import bpy

import numpy as np
from pathlib import Path
from typing import List
import random

from constants.directories import HDRI_PURE_SKIES_DIRECTORY
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

HDRI_EXTENSIONS = [".exr", ".hdr"]


def get_all_hdri_by_directory(directory: Path = HDRI_PURE_SKIES_DIRECTORY) -> List[str]:
    """
    Get all HDRI files in the specified directory.

    Args:
        directory: The directory to search for HDRI files.

    Returns:
        A list of HDRI file paths.
    """
    hdri_files = []

    for ext in HDRI_EXTENSIONS:
        hdri_files += list(directory.glob(f"*{ext}"))

    if not hdri_files:
        logger.error(f"No HDRI files found in {directory}")
        raise

    logger.info(f"Found {len(hdri_files)} HDRI files in {directory}")

    return hdri_files


def add_sky_to_scene(directory: Path = HDRI_PURE_SKIES_DIRECTORY):
    hdri_paths = get_all_hdri_by_directory(directory)
    random_hdri_path = random.choice(hdri_paths)

    node_tree = bpy.context.scene.world.node_tree
    tree_nodes = node_tree.nodes
    tree_nodes.clear()

    add_hdri(random_hdri_path, tree_nodes, node_tree)
    add_sky_texture(tree_nodes, node_tree)

    node_add = tree_nodes.new(type="ShaderNodeAddShader")  # add shader node
    tree_nodes = node_tree.nodes
    k = 0
    for node in tree_nodes:
        if "Background" in node.name:
            links = node_tree.links
            _ = links.new(node.outputs["Background"], node_add.inputs[k])
            k += 1

    node_output = tree_nodes.new(type="ShaderNodeOutputWorld")
    _ = links.new(node_add.outputs["Shader"], node_output.inputs["Surface"])


def add_hdri(path: Path, tree_nodes, node_tree):
    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_background.inputs["Strength"].default_value = clamp(random.uniform(0.6, 1.0), 0.6, 1.0)

    node_environment = tree_nodes.new("ShaderNodeTexEnvironment")
    node_environment.image = bpy.data.images.load(path.as_posix())

    node_blackbody = tree_nodes.new("ShaderNodeBlackbody")
    node_blackbody.inputs["Temperature"].default_value = random.randint(5000, 6500)

    node_multiply = tree_nodes.new("ShaderNodeVectorMath")
    node_multiply.operation = "MULTIPLY"

    links = node_tree.links

    _ = links.new(node_multiply.outputs["Vector"], node_background.inputs["Color"])

    _ = links.new(node_blackbody.outputs["Color"], node_multiply.inputs[0])
    _ = links.new(node_environment.outputs["Color"], node_multiply.inputs[1])


def add_sky_texture(tree_nodes, node_tree):
    node_sky = tree_nodes.new(type="ShaderNodeTexSky")
    node_sky.sky_type = "NISHITA"
    node_sky.sun_size = np.deg2rad(random.uniform(1, 3))

    node_sky.sun_elevation = np.deg2rad(random.uniform(45, 90))
    node_sky.sun_rotation = np.deg2rad(random.uniform(0, 360))
    node_sky.sun_intensity = random.uniform(0.4, 0.8)

    node_sky.altitude = random.randint(0, 100)
    node_sky.air_density = clamp(random.randint(0, 2), 0, 2)
    node_sky.dust_density = clamp(random.randint(0, 2), 0, 2)
    node_sky.ozone_density = clamp(random.randint(0, 2), 0, 2)

    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_background.inputs["Strength"].default_value = clamp(
        random.uniform(0.2, 0.6), 0.0, 1.0
    )

    links = node_tree.links
    _ = links.new(node_sky.outputs["Color"], node_background.inputs["Color"])


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)
