import bpy
import numpy as np
from pathlib import Path
from typing import List
import random
import logging

from configuration.configuration import Configuration
from configuration.sky import SkyConfiguration, SunConfiguration
from constants.file_extensions import FileExtension
from utils.metadata import add_entry, MetadataKey


class Constants:
    HDRI_EXTENSIONS: List[str] = [f".{FileExtension.EXR.value}", f".{FileExtension.HDR.value}"]
    MIN: str = "min"
    MAX: str = "max"

    # Shader node types
    SHADER_NODE_BACKGROUND: str = "ShaderNodeBackground"
    SHADER_NODE_BLACKBODY: str = "ShaderNodeBlackbody"
    SHADER_NODE_TEX_ENVIRONMENT: str = "ShaderNodeTexEnvironment"
    SHADER_NODE_TEX_SKY: str = "ShaderNodeTexSky"
    SHADER_NODE_VECTOR_MATH: str = "ShaderNodeVectorMath"
    SHADER_NODE_MIX_SHADER: str = "ShaderNodeMixShader"
    SHADER_NODE_OUTPUT_WORLD: str = "ShaderNodeOutputWorld"

    # Node properties
    NODE_TYPE_BACKGROUND: str = "BACKGROUND"
    BACKGROUND: str = "Background"
    COLOR: str = "Color"
    STRENGTH: str = "Strength"
    TEMPERATURE: str = "Temperature"
    VECTOR: str = "Vector"
    MULTIPLY: str = "MULTIPLY"
    SHADER: str = "Shader"
    SURFACE: str = "Surface"
    FAC: str = "Fac"


def get_all_hdri_by_directory(
        directory: str,
        include: List[str] | None = None,
        exclude: List[str] | None = None,
) -> List[Path]:
    """
    Retrieve all HDRI files in the specified directory.

    Args:
        directory (Path): The directory to search for HDRI files.
        include (List[str], optional): A list of keywords that must be present in the file path. Defaults to None.
        exclude (List[str], optional): A list of keywords to filter out from the file paths. Defaults to None.

    Returns:
        List[Path]: A list of HDRI file paths.

    Raises:
        FileNotFoundError: If no HDRI files are found in the directory.
    """
    directory = Path(directory)

    hdri_files = []
    for ext in Constants.HDRI_EXTENSIONS:
        hdri_files += list(directory.glob(f"*{ext}"))

    if include:
        hdri_files = [path for path in hdri_files if any(keyword in path.stem for keyword in include)]

    if exclude:
        hdri_files = [path for path in hdri_files if not any(keyword in path.stem for keyword in exclude)]

    if not hdri_files:
        logging.error(f"No HDRI files found in {directory}")
        raise FileNotFoundError(f"No HDRI files found in {directory}")

    logging.debug(f"Found {len(hdri_files)} HDRI files in {directory}")
    return hdri_files


def add_sky_to_scene(configuration: Configuration, seed: int = None) -> None:
    """
    Add a random sky (HDRI or sky texture) to the Blender scene.

    Args:
        configuration (Configuration): The HDRI configuration.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    sky_configuration = configuration.sky_configuration
    directory = configuration.sky_configuration.directory

    logging.debug(f"Adding sky to the scene from directory: {directory}")
    hdri_paths = get_all_hdri_by_directory(
        directory=directory,
        include=sky_configuration.include,
        exclude=sky_configuration.exclude
    )
    random_hdri_path = random.choice(hdri_paths)
    add_entry(category=MetadataKey.HDRIS, filepath=random_hdri_path.as_posix())
    logging.info(f"Selected HDRI file: {random_hdri_path}")

    node_tree = bpy.context.scene.world.node_tree
    tree_nodes = node_tree.nodes
    logging.debug("Clearing existing nodes from the world node tree.")
    tree_nodes.clear()

    logging.info("Adding HDRI to the scene...")
    _add_hdri(sky_configuration, random_hdri_path, tree_nodes, node_tree)
    logging.info("HDRI added to the scene successfully.")

    logging.info("Adding procedural sky texture to the scene...")
    _add_sky_texture(sky_configuration, tree_nodes, node_tree, seed)
    logging.info("Procedural sky texture added to the scene successfully.")

    logging.info("Setting up world output shader...")
    _setup_world_output(tree_nodes, node_tree)
    logging.info("World output shader set up successfully.")


def _add_hdri(
        configuration: SkyConfiguration,
        path: Path,
        tree_nodes: bpy.types.bpy_prop_collection,
        node_tree: bpy.types.ShaderNodeTree
) -> None:
    """
    Add HDRI node setup to the node tree.

    Args:
        configuration (SkyConfiguration): The configuration for HDRI settings.
        path (Path): The path to the HDRI file.
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        node_tree (bpy.types.ShaderNodeTree): The node tree to which the HDRI nodes will be added.
    """
    logging.debug(f"Adding HDRI node for {path}")

    node_background = _create_background_node(tree_nodes, configuration)
    node_environment = _create_environment_node(tree_nodes, path)
    node_blackbody = _create_blackbody_node(tree_nodes, configuration)
    node_multiply = _create_vector_math_node(tree_nodes)

    _link_hdri_nodes(node_tree, node_background, node_environment, node_blackbody, node_multiply)


def _add_sky_texture(
        configuration: SkyConfiguration,
        tree_nodes: bpy.types.bpy_prop_collection,
        node_tree: bpy.types.ShaderNodeTree,
        seed: int = None
) -> None:
    """
    Add a procedural sky texture node setup to the node tree.

    Args:
        configuration (SkyConfiguration): The configuration for sky settings.
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        node_tree (bpy.types.ShaderNodeTree): The node tree to which the sky texture nodes will be added.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    logging.debug("Adding procedural sky texture.")
    if seed is not None:
        random.seed(seed)
        logging.debug(f"Seed set to {seed}")

    sun_config = configuration.sun_configuration

    node_sky = _create_sky_texture_node(tree_nodes, sun_config, configuration)
    node_background = _create_sky_background_node(tree_nodes)

    _link_sky_texture_nodes(node_tree, node_sky, node_background)


def _create_sky_texture_node(
        tree_nodes: bpy.types.bpy_prop_collection,
        sun_config: SunConfiguration,
        configuration: SkyConfiguration
) -> bpy.types.ShaderNodeTexSky:
    """
    Create a procedural sky texture node.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        sun_config (SunConfiguration): The sun configuration for the sky texture.
        configuration (SkyConfiguration): The configuration for sky settings.

    Returns:
        bpy.types.ShaderNodeTexSky: The created sky texture node.
    """
    node_sky = tree_nodes.new(type=Constants.SHADER_NODE_TEX_SKY)

    node_sky.sky_type = configuration.sky_type
    node_sky.sun_size = np.deg2rad(random.uniform(sun_config.size[Constants.MIN], sun_config.size[Constants.MAX]))
    node_sky.sun_elevation = np.deg2rad(
        random.uniform(sun_config.elevation[Constants.MIN], sun_config.elevation[Constants.MAX]))
    node_sky.sun_rotation = np.deg2rad(
        random.uniform(sun_config.rotation[Constants.MIN], sun_config.rotation[Constants.MAX]))
    node_sky.sun_intensity = random.uniform(sun_config.intensity[Constants.MIN], sun_config.intensity[Constants.MAX])

    logging.debug(f"Sky texture parameters - Sun size: {node_sky.sun_size}, Elevation: {node_sky.sun_elevation}, "
                  f"Rotation: {node_sky.sun_rotation}, Intensity: {node_sky.sun_intensity}")

    node_sky.altitude = random.randint(configuration.density[Constants.MIN], configuration.density[Constants.MAX])
    node_sky.air_density = _clamp(
        random.randint(configuration.density[Constants.MIN], configuration.density[Constants.MAX]),
        configuration.density[Constants.MIN], configuration.density[Constants.MAX])
    node_sky.dust_density = _clamp(
        random.randint(configuration.density[Constants.MIN], configuration.density[Constants.MAX]),
        configuration.density[Constants.MIN], configuration.density[Constants.MAX])
    node_sky.ozone_density = _clamp(
        random.randint(configuration.density[Constants.MIN], configuration.density[Constants.MAX]),
        configuration.density[Constants.MIN], configuration.density[Constants.MAX])
    logging.debug(
        f"Sky texture atmospheric values - Altitude: {node_sky.altitude}, Air density: {node_sky.air_density}, "
        f"Dust density: {node_sky.dust_density}, Ozone density: {node_sky.ozone_density}")

    return node_sky


def _create_sky_background_node(tree_nodes: bpy.types.bpy_prop_collection) -> bpy.types.ShaderNodeBackground:
    """
    Create a background node for the sky texture.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.

    Returns:
        bpy.types.ShaderNodeBackground: The created background node.
    """
    node_background = tree_nodes.new(type=Constants.SHADER_NODE_BACKGROUND)
    node_background.inputs[Constants.STRENGTH].default_value = _clamp(random.uniform(0.2, 0.6), 0.0, 1.0)

    logging.debug(f'Set background strength to {node_background.inputs[Constants.STRENGTH].default_value}')

    return node_background


def _link_sky_texture_nodes(
        node_tree: bpy.types.ShaderNodeTree,
        node_sky: bpy.types.ShaderNodeTexSky,
        node_background: bpy.types.ShaderNodeBackground
) -> None:
    """
    Link the sky texture nodes together in the node tree.

    Args:
        node_tree (bpy.types.ShaderNodeTree): The node tree of the scene.
        node_sky (bpy.types.ShaderNodeTexSky): The sky texture node.
        node_background (bpy.types.ShaderNodeBackground): The background node.
    """
    links = node_tree.links
    links.new(node_sky.outputs[Constants.COLOR], node_background.inputs[Constants.COLOR])

    logging.debug("Linked sky texture nodes successfully.")


def _create_background_node(
        tree_nodes: bpy.types.bpy_prop_collection,
        configuration: SkyConfiguration
) -> bpy.types.ShaderNodeBackground:
    """
    Create a background node with strength from configuration.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        configuration (SkyConfiguration): The configuration for HDRI settings.

    Returns:
        bpy.types.ShaderNodeBackground: The created background node.
    """
    node_background = tree_nodes.new(type=Constants.SHADER_NODE_BACKGROUND)
    node_background.inputs[Constants.STRENGTH].default_value = _clamp(
        random.uniform(configuration.strength[Constants.MIN], configuration.strength[Constants.MAX]),
        configuration.strength[Constants.MIN], configuration.strength[Constants.MAX]
    )

    logging.debug(f'Set background strength to {node_background.inputs[Constants.STRENGTH].default_value}')

    return node_background


def _create_environment_node(
        tree_nodes: bpy.types.bpy_prop_collection,
        path: Path
) -> bpy.types.ShaderNodeTexEnvironment:
    """
    Create an environment texture node from an HDRI file.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        path (Path): The path to the HDRI file.

    Returns:
        bpy.types.ShaderNodeTexEnvironment: The created environment node.
    """
    node_environment = tree_nodes.new(Constants.SHADER_NODE_TEX_ENVIRONMENT)
    node_environment.image = bpy.data.images.load(path.as_posix())

    logging.debug("Loaded HDRI image.")

    return node_environment


def _create_blackbody_node(
        tree_nodes: bpy.types.bpy_prop_collection,
        configuration: SkyConfiguration
) -> bpy.types.ShaderNodeBlackbody:
    """
    Create a blackbody node with temperature from configuration.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        configuration (SkyConfiguration): The configuration for HDRI settings.

    Returns:
        bpy.types.ShaderNodeBlackbody: The created blackbody node.
    """
    node_blackbody = tree_nodes.new(Constants.SHADER_NODE_BLACKBODY)
    node_blackbody.inputs[Constants.TEMPERATURE].default_value = random.randint(
        configuration.temperature[Constants.MIN], configuration.temperature[Constants.MAX]
    )

    logging.debug(f'Set blackbody temperature to {node_blackbody.inputs[Constants.TEMPERATURE].default_value}')

    return node_blackbody


def _create_vector_math_node(tree_nodes: bpy.types.bpy_prop_collection) -> bpy.types.ShaderNodeVectorMath:
    """
    Create a vector math node with 'MULTIPLY' operation.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.

    Returns:
        bpy.types.ShaderNodeVectorMath: The created vector math node.
    """
    node_multiply = tree_nodes.new(Constants.SHADER_NODE_VECTOR_MATH)
    node_multiply.operation = Constants.MULTIPLY

    logging.debug("Added multiply operation for vector math.")

    return node_multiply


def _link_hdri_nodes(
        node_tree: bpy.types.ShaderNodeTree,
        node_background: bpy.types.ShaderNodeBackground,
        node_environment: bpy.types.ShaderNodeTexEnvironment,
        node_blackbody: bpy.types.ShaderNodeBlackbody,
        node_multiply: bpy.types.ShaderNodeVectorMath
) -> None:
    """
    Link the HDRI nodes together in the node tree.

    Args:
        node_tree (bpy.types.ShaderNodeTree): The node tree of the scene.
        node_background (bpy.types.ShaderNodeBackground): The background node.
        node_environment (bpy.types.ShaderNodeTexEnvironment): The environment node.
        node_blackbody (bpy.types.ShaderNodeBlackbody): The blackbody node.
        node_multiply (bpy.types.ShaderNodeVectorMath): The vector math node.
    """
    links = node_tree.links

    links.new(node_multiply.outputs[Constants.VECTOR], node_background.inputs[Constants.COLOR])
    links.new(node_blackbody.outputs[Constants.COLOR], node_multiply.inputs[0])
    links.new(node_environment.outputs[Constants.COLOR], node_multiply.inputs[1])

    logging.debug("Linked HDRI nodes successfully.")


def _setup_world_output(tree_nodes: bpy.types.bpy_prop_collection, node_tree: bpy.types.ShaderNodeTree) -> None:
    """
    Set up the output shader for the world background in the node tree.

    Args:
        tree_nodes (bpy.types.bpy_prop_collection): The tree nodes of the scene.
        node_tree (bpy.types.ShaderNodeTree): The node tree to which the output shader will be added.
    """
    logging.debug("Setting up world output shader using Mix Shader.")

    # Create a Mix Shader node instead of an Add Shader
    mix_node = tree_nodes.new(type=Constants.SHADER_NODE_MIX_SHADER)
    links = node_tree.links

    # Find the two background nodes by checking the node type
    bg_hdri = None
    bg_sky = None
    for node in tree_nodes:
        if node.type == Constants.NODE_TYPE_BACKGROUND:
            if bg_hdri is None:
                bg_hdri = node
            else:
                bg_sky = node

    # Ensure both background nodes are found
    if bg_hdri is None or bg_sky is None:
        missing = []
        if bg_hdri is None:
            missing.append("bg_hdri")
        if bg_sky is None:
            missing.append("bg_sky")
        logging.error(
            f"Not enough background nodes found to set up the world output shader. Missing: {', '.join(missing)}.")
        return

    # Set a factor for mixing
    mix_factor = 0.5
    mix_node.inputs[Constants.FAC].default_value = mix_factor

    # Link the two backgrounds to the Mix Shader
    links.new(bg_hdri.outputs[Constants.BACKGROUND], mix_node.inputs[1])
    links.new(bg_sky.outputs[Constants.BACKGROUND], mix_node.inputs[2])

    # Create the World Output node and connect the Mix Shader output to it
    node_output = tree_nodes.new(type=Constants.SHADER_NODE_OUTPUT_WORLD)
    links.new(mix_node.outputs[Constants.SHADER], node_output.inputs[Constants.SURFACE])

    logging.debug("World output shader set up successfully.")


def _clamp(value: float, min_value: float, max_value: float) -> float:
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
