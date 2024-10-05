import bpy

from pathlib import Path

from configuration.configuration import RenderType, RenderConfiguration
from configuration.consts import Constants
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


class RenderingOutputsConstants:
    class TreeNode:
        RENDER_LAYERS = "Render Layers"
        FILE_OUTPUT = "File Output"

    class Compositor:
        NODE_OUTPUT_FILE = "CompositorNodeOutputFile"
        NODE_ID_MASK = "CompositorNodeIDMask"

    class FileSlot:
        IMAGE = "Image"
        ID_MASK = "IDMask"


def setup_outputs(
        scene: bpy.context.scene,
        render_configuration: RenderConfiguration,
        render_image: bool = True,
        render_object_index: bool = True,
        output_path: Path = Constants.Directory.OUTPUT_DIR,
        output_name: str = None
) -> None:
    logger.info("Setting up rendering outputs...")

    view_layer: bpy.types.ViewLayer = scene.view_layers[0]

    view_layer.use_pass_object_index = render_object_index and render_configuration.render == RenderType.Cycles
    view_layer.use_pass_material_index = False
    view_layer.use_pass_normal = False
    view_layer.use_pass_z = False

    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.CompositorNodeTree = scene.node_tree
    render_layers: bpy.types.CompositorNodeRLayers = node_tree.nodes.get(
        RenderingOutputsConstants.TreeNode.RENDER_LAYERS
    )

    node_tree.nodes.new(type=RenderingOutputsConstants.Compositor.NODE_OUTPUT_FILE)
    output_file_node: bpy.types.CompositorNodeOutputFile = node_tree.nodes.get(
        RenderingOutputsConstants.TreeNode.FILE_OUTPUT
    )
    output_file_node.inputs.clear()
    output_file_node.base_path = output_path.as_posix()

    if render_image:
        _setup_image_output(node_tree, output_file_node, render_layers, output_name)

    if view_layer.use_pass_object_index:
        map_range_node_indexOB = node_tree.nodes.new(type="CompositorNodeMapRange")
        map_range_node_indexOB.inputs["From Min"].default_value = 0
        map_range_node_indexOB.inputs["From Max"].default_value = 255
        map_range_node_indexOB.inputs["To Min"].default_value = 0
        map_range_node_indexOB.inputs["To Max"].default_value = 1

        output_file_node.file_slots.new("IndexOB")
        output = output_file_node.file_slots["IndexOB"]
        print(output)

        output.use_node_format = False
        output.format.file_format = "PNG"
        output.format.color_mode = "BW"
        output.path = f"{output_name}_IndexOB" if output_name else "IndexOB"

        indexOB_output = render_layers.outputs.get("IndexOB")
        print(indexOB_output)
        print(map_range_node_indexOB.inputs["Value"])

        _ = node_tree.links.new(indexOB_output, map_range_node_indexOB.inputs["Value"])
        _ = node_tree.links.new(
            map_range_node_indexOB.outputs["Value"], output_file_node.inputs["IndexOB"]
        )

        _setup_mask_output(node_tree, output_file_node, render_layers, output_name)

    logger.info(f"Configured the following outputs:", extra={
        "Render Image": render_image,
        "Render Object Index": view_layer.use_pass_object_index,
    })


def _setup_image_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        output_name: str = None,
) -> None:
    image_file_slot_title = RenderingOutputsConstants.FileSlot.IMAGE

    # Create the image output slot
    output_file_node.file_slots.new(image_file_slot_title)
    file_output_image_slot = bpy.types.NodeOutputFileSlotFile = output_file_node.file_slots[image_file_slot_title]

    # Configure the image output
    file_output_image_slot.use_node_format = False  # Custom format
    file_output_image_slot.format.file_format = Constants.Render.FILE_FORMAT
    file_output_image_slot.format.color_mode = Constants.Render.COLOR_MODE

    file_output_image_slot.path = f"{output_name}_{image_file_slot_title}" if output_name else f"{image_file_slot_title}"

    image_output: bpy.types.NodeSocketColor = render_layers.outputs.get(image_file_slot_title)
    if image_output is None:
        logger.error(f"The '{image_file_slot_title}' output could not be found")
        return

    image_input: bpy.types.NodeSocketColor = output_file_node.inputs[image_file_slot_title]
    if image_input is None:
        logger.error(f"The '{image_file_slot_title}' input could not be found")
        return

    # Link the image output to the file output node
    _ = node_tree.links.new(image_output, image_input)


def _setup_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        output_name: str = None,
) -> None:
    id_mask_node = node_tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = 255
    id_mask_node.use_antialiasing = True

    output_file_node.file_slots.new("IDMask")

    output_file_node.file_slots["IDMask"].use_node_format = False  # custom format
    output_file_node.file_slots["IDMask"].format.file_format = "PNG"
    output_file_node.file_slots["IDMask"].format.color_mode = "BW"

    output_file_node.file_slots["IDMask"].path = f"{output_name}_IDMask" if output_name else "IDMask"

    _ = node_tree.links.new(render_layers.outputs.get("IndexOB"), id_mask_node.inputs["ID value"])
    _ = node_tree.links.new(id_mask_node.outputs["Alpha"], output_file_node.inputs["IDMask"])
