import bpy
from pathlib import Path
from configuration.configuration import EngineType, RenderConfiguration
from constants.directories import OUTPUT_DIRECTORY
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def setup_outputs(
        scene: bpy.context.scene,
        render_configuration: RenderConfiguration,
        render_image: bool = True,
        render_object_index: bool = True,
        render_environment: bool = True,
        output_path: Path = OUTPUT_DIRECTORY
) -> None:
    """
    Set up rendering outputs.

    Args:
        scene (bpy.context.scene): The Blender scene object.
        render_configuration (RenderConfiguration): The render configuration settings.
        render_image (bool, optional): Whether to render the image. Defaults to True.
        render_object_index (bool, optional): Whether to render the object index. Defaults to True.
        render_environment (bool, optional): Whether to render the environment. Defaults to False.
        output_path (Path, optional): The output path for the rendered files. Defaults to OUTPUT_DIRECTORY.
    """
    logger.info("Setting up rendering outputs...")

    view_layer: bpy.types.ViewLayer = scene.view_layers[0]

    # Configure rendering passes based on the configuration
    view_layer.use_pass_object_index = render_object_index and render_configuration.engine == EngineType.Cycles
    view_layer.use_pass_environment = render_environment and render_configuration.engine == EngineType.Cycles
    view_layer.use_pass_material_index = False
    view_layer.use_pass_normal = False
    view_layer.use_pass_z = False
    logger.debug(
        f"Passes configured - Object Index: {view_layer.use_pass_object_index}, Material Index: {view_layer.use_pass_material_index}")

    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.CompositorNodeTree = scene.node_tree
    render_layers: bpy.types.CompositorNodeRLayers = node_tree.nodes.get("Render Layers")

    output_file_node: bpy.types.CompositorNodeOutputFile = node_tree.nodes.get("File Output") or node_tree.nodes.new(
        type="CompositorNodeOutputFile")
    output_file_node.inputs.clear()
    output_file_node.base_path = output_path.as_posix()
    logger.debug(f"Output file node set with base path: {output_path.as_posix()}")

    if render_image:
        _setup_image_output(node_tree, output_file_node, render_layers)

    if view_layer.use_pass_object_index:
        _setup_object_index_output(node_tree, output_file_node, render_layers)
        _setup_biome_mask_output(node_tree, output_file_node, render_layers)

    if view_layer.use_pass_environment:
        # Enable compositing for environment pass
        scene.render.use_compositing = True

        _setup_environment_mask_output(node_tree, output_file_node, render_layers)

    logger.info("Render outputs configured.", extra={
        "Render Image": render_image,
        "Render Object Index": view_layer.use_pass_object_index,
        "Render Environment": view_layer.use_pass_environment
    })


def _setup_image_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the image output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up image output.")
    output_file_node.file_slots.new("Image")
    image_file_slot = output_file_node.file_slots["Image"]
    image_file_slot.use_node_format = False  # Custom format
    image_file_slot.format.file_format = "PNG"
    image_file_slot.format.color_mode = "RGBA"
    image_file_slot.path = "Image"

    image_output = render_layers.outputs.get("Image")
    if image_output:
        _ = node_tree.links.new(image_output, output_file_node.inputs["Image"])
        logger.info("Linked image output to the file node.")
    else:
        logger.error("Render Layers node does not contain an 'Image' output.")


def _setup_object_index_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the object index output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up object index output.")
    map_range_node_indexOB = node_tree.nodes.new(type="CompositorNodeMapRange")
    map_range_node_indexOB.inputs["From Min"].default_value = 0
    map_range_node_indexOB.inputs["From Max"].default_value = 255
    map_range_node_indexOB.inputs["To Min"].default_value = 0
    map_range_node_indexOB.inputs["To Max"].default_value = 1

    output_file_node.file_slots.new("IndexOB")
    index_ob_file_slot = output_file_node.file_slots["IndexOB"]
    index_ob_file_slot.use_node_format = False
    index_ob_file_slot.format.file_format = "PNG"
    index_ob_file_slot.format.color_mode = "BW"
    index_ob_file_slot.path = "IndexOB"

    index_ob_output = render_layers.outputs.get("IndexOB")
    if index_ob_output:
        _ = node_tree.links.new(index_ob_output, map_range_node_indexOB.inputs["Value"])
        _ = node_tree.links.new(map_range_node_indexOB.outputs["Value"], output_file_node.inputs["IndexOB"])
        logger.info("Linked object index output to the file node.")
    else:
        logger.error("Render Layers node does not contain an 'IndexOB' output.")


def _setup_biome_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the biome mask output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up terrain mask output.")
    id_mask_node = node_tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = 255
    id_mask_node.use_antialiasing = True

    output_file_node.file_slots.new("BiomeMask")
    id_mask_file_slot = output_file_node.file_slots["BiomeMask"]
    id_mask_file_slot.use_node_format = False  # Custom format
    id_mask_file_slot.format.file_format = "PNG"
    id_mask_file_slot.format.color_mode = "BW"
    id_mask_file_slot.path = "BiomeMask"

    id_mask_output = render_layers.outputs.get("IndexOB")
    if id_mask_output:
        _ = node_tree.links.new(id_mask_output, id_mask_node.inputs["ID value"])
        _ = node_tree.links.new(id_mask_node.outputs["Alpha"], output_file_node.inputs["BiomeMask"])
        logger.info("Linked biome mask output to the file node.")
    else:
        logger.error("Render Layers node does not contain an 'IndexOB' output for biome mask.")


def _setup_environment_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the environment mask output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up environment mask output.")

    output_file_node.file_slots.new("HDRIMask")
    hdri_mask_file_slot = output_file_node.file_slots["HDRIMask"]
    hdri_mask_file_slot.use_node_format = False  # Custom format
    hdri_mask_file_slot.format.file_format = "PNG"
    hdri_mask_file_slot.format.color_mode = "BW"  # Black and white for mask output
    hdri_mask_file_slot.path = "HDRIMask"

    env_output = render_layers.outputs.get("Env")
    if env_output:
        # Link the 'Env' output directly to the file output node
        _ = node_tree.links.new(env_output, output_file_node.inputs["HDRIMask"])
        logger.info("Linked environment mask output to the file node.")
    else:
        logger.error("Render Layers node does not contain an 'Env' output.")
