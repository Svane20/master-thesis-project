import bpy

from pathlib import Path

from configuration.configuration import RenderType, RenderConfiguration
from configuration.consts import Constants
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def setup_outputs(
        scene: bpy.context.scene,
        render_configuration: RenderConfiguration,
        render_image: bool = True,
        render_object_index: bool = True,
        output_path: Path = Constants.Directory.OUTPUT_DIR
) -> None:
    """
    Set up rendering outputs.

    Args:
        scene: The scene.
        render_configuration: The render configuration.
        render_image: Whether to render the image.
        render_object_index: Whether to render the object index.
        output_path: The output path.
    """

    logger.info("Setting up rendering outputs...")

    view_layer: bpy.types.ViewLayer = scene.view_layers[0]

    view_layer.use_pass_object_index = render_object_index and render_configuration.render == RenderType.Cycles
    view_layer.use_pass_material_index = False
    view_layer.use_pass_normal = False
    view_layer.use_pass_z = False

    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.CompositorNodeTree = scene.node_tree
    render_layers: bpy.types.CompositorNodeRLayers = node_tree.nodes.get("Render Layers")

    node_tree.nodes.new(type="CompositorNodeOutputFile")
    output_file_node: bpy.types.CompositorNodeOutputFile = node_tree.nodes.get("File Output")
    output_file_node.inputs.clear()
    output_file_node.base_path = output_path.as_posix()

    if render_image:
        _setup_image_output(node_tree, output_file_node, render_layers)

    if view_layer.use_pass_object_index:
        _setup_object_index_output(node_tree, output_file_node, render_layers)
        _setup_id_mask_output(node_tree, output_file_node, render_layers)

    logger.info(f"Configured the following outputs:", extra={
        "Render Image": render_image,
        "Render Object Index": view_layer.use_pass_object_index,
    })


def _setup_image_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the image output.

    Args:
        node_tree: The node tree.
        output_file_node: The output file node.
        render_layers: The render layers.
    """
    output_file_node.file_slots.new("Image")
    image_file_slot = output_file_node.file_slots["Image"]

    image_file_slot.use_node_format = False  # custom format
    image_file_slot.format.file_format = "PNG"
    image_file_slot.format.color_mode = "RGBA"
    image_file_slot.path = "Image"

    image_output = render_layers.outputs.get("Image")
    _ = node_tree.links.new(image_output, output_file_node.inputs["Image"])


def _setup_object_index_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the object index output.

    Args:
        node_tree: The node tree.
        output_file_node: The output file node.
        render_layers: The render layers.
    """
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
    index_ob_input = output_file_node.inputs["IndexOB"]

    _ = node_tree.links.new(index_ob_output, map_range_node_indexOB.inputs["Value"])
    _ = node_tree.links.new(map_range_node_indexOB.outputs["Value"], index_ob_input)


def _setup_id_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
) -> None:
    """
    Set up the ID mask output.

    Args:
        node_tree: The node tree.
        output_file_node: The output file node.
        render_layers: The render layers.
    """
    id_mask_node = node_tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = 255
    id_mask_node.use_antialiasing = True

    output_file_node.file_slots.new("IDMask")
    id_mask_file_slot = output_file_node.file_slots["IDMask"]

    id_mask_file_slot.use_node_format = False  # Custom format
    id_mask_file_slot.format.file_format = "PNG"
    id_mask_file_slot.format.color_mode = "BW"
    id_mask_file_slot.path = "IDMask"

    id_mask_output = render_layers.outputs.get("IndexOB")
    id_mask_input = output_file_node.inputs["IDMask"]

    _ = node_tree.links.new(id_mask_output, id_mask_node.inputs["ID value"])
    _ = node_tree.links.new(id_mask_node.outputs["Alpha"], id_mask_input)
