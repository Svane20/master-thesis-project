import bpy

from configuration.outputs import OutputsConfiguration, NodeOutputConfiguration
from configuration.render import EngineType
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

RENDER_LAYERS = "Render Layers"
FILE_OUTPUT = "File Output"

COMPOSITOR_NODE_MAP_RANGE = "CompositorNodeMapRange"
COMPOSITOR_NODE_ID_MASK = "CompositorNodeIDMask"
COMPOSITOR_NODE_OUTPUT_FILE = "CompositorNodeOutputFile"

FROM_MIN = "From Min"
FROM_MAX = "From Max"
TO_MIN = "To Min"
TO_MAX = "To Max"
VALUE = "Value"
ID_VALUE = "ID value"
ALPHA = "Alpha"
ENV = "Env"


def setup_outputs(
        scene: bpy.context.scene,
        engine_type: str,
        outputs_configuration: OutputsConfiguration,
        render_image: bool = True,
        render_object_index: bool = True,
        render_environment: bool = True,
        output_path: str = str
) -> None:
    """
    Set up rendering outputs for image, object index, and environment masks.

    Args:
        scene (bpy.context.scene): The Blender scene object.
        engine_type (EngineType): The render engine type.
        outputs_configuration (OutputsConfiguration): The outputs configuration settings.
        render_image (bool, optional): Whether to render the image. Defaults to True.
        render_object_index (bool, optional): Whether to render the object index. Defaults to True.
        render_environment (bool, optional): Whether to render the environment. Defaults to True.
        output_path (Path, optional): The output path for the rendered files. Defaults to OUTPUT_DIRECTORY.
    """
    logger.info("Setting up rendering outputs...")

    view_layer: bpy.types.ViewLayer = scene.view_layers[0]

    # Configure rendering passes based on the configuration
    view_layer.use_pass_object_index = render_object_index and engine_type == EngineType.Cycles.value
    view_layer.use_pass_environment = render_environment and engine_type == EngineType.Cycles.value
    view_layer.use_pass_material_index = False
    view_layer.use_pass_normal = False
    view_layer.use_pass_z = False

    logger.debug(f"Passes configured", extra={
        "Object Index": view_layer.use_pass_object_index,
        "Environment": view_layer.use_pass_environment
    })

    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.CompositorNodeTree = scene.node_tree
    render_layers: bpy.types.CompositorNodeRLayers = node_tree.nodes.get(RENDER_LAYERS)

    output_file_node: bpy.types.CompositorNodeOutputFile = node_tree.nodes.get(FILE_OUTPUT) or node_tree.nodes.new(
        type=COMPOSITOR_NODE_OUTPUT_FILE)
    output_file_node.inputs.clear()
    output_file_node.base_path = output_path
    logger.debug(f"Output file node set to: {output_file_node.base_path}")

    if render_image:
        _setup_image_output(
            node_tree,
            output_file_node,
            render_layers,
            image_output_configuration=outputs_configuration.image_output_configuration
        )

    if view_layer.use_pass_object_index:
        _setup_object_index_output(
            node_tree,
            output_file_node,
            render_layers,
            object_index_output_configuration=outputs_configuration.object_index_output_configuration
        )
        _setup_id_mask_output(
            node_tree,
            output_file_node,
            render_layers,
            object_index_output_configuration=outputs_configuration.object_index_output_configuration,
            id_mask_output_configuration=outputs_configuration.id_mask_output_configuration
        )

    if view_layer.use_pass_environment:
        # Enable compositing for environment pass
        scene.render.use_compositing = True

        _setup_environment_mask_output(
            node_tree,
            output_file_node,
            render_layers,
            environment_output_configuration=outputs_configuration.environment_output_configuration
        )

    logger.info("Render outputs configured.", extra={
        "Render Image": render_image,
        "Render Object Index": view_layer.use_pass_object_index,
        "Render Environment": view_layer.use_pass_environment
    })


def _setup_image_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        image_output_configuration: NodeOutputConfiguration
) -> None:
    """
    Set up the image output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up image output.")
    image_title = image_output_configuration.title

    output_file_node.file_slots.new(image_title)
    image_file_slot = output_file_node.file_slots[image_title]
    image_file_slot.use_node_format = image_output_configuration.use_node_format
    image_file_slot.format.file_format = image_output_configuration.file_format
    image_file_slot.format.color_mode = image_output_configuration.color_mode
    image_file_slot.path = image_output_configuration.path

    image_output = render_layers.outputs.get(image_title)
    if image_output:
        _ = node_tree.links.new(image_output, output_file_node.inputs[image_title])
        logger.info(f"Linked {image_title} output to the file node.")
    else:
        logger.error(f"Render Layers node does not contain an '{image_title}' output.")


def _setup_object_index_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        object_index_output_configuration: NodeOutputConfiguration
) -> None:
    """
    Set up the object index output for rendering.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
        object_index_output_configuration (NodeOutputConfiguration): The configuration for the object index output.
    """
    logger.debug("Setting up object index output.")
    map_range_node_indexOB = node_tree.nodes.new(type=COMPOSITOR_NODE_MAP_RANGE)
    map_range_node_indexOB.inputs[FROM_MIN].default_value = 0
    map_range_node_indexOB.inputs[FROM_MAX].default_value = 255
    map_range_node_indexOB.inputs[TO_MIN].default_value = 0
    map_range_node_indexOB.inputs[TO_MAX].default_value = 1

    index_ob_title = object_index_output_configuration.title

    output_file_node.file_slots.new(index_ob_title)
    index_ob_file_slot = output_file_node.file_slots[index_ob_title]
    index_ob_file_slot.use_node_format = object_index_output_configuration.use_node_format
    index_ob_file_slot.format.file_format = object_index_output_configuration.file_format
    index_ob_file_slot.format.color_mode = object_index_output_configuration.color_mode
    index_ob_file_slot.path = object_index_output_configuration.path

    index_ob_output = render_layers.outputs.get(index_ob_title)
    if index_ob_output:
        _ = node_tree.links.new(index_ob_output, map_range_node_indexOB.inputs[VALUE])
        _ = node_tree.links.new(map_range_node_indexOB.outputs[VALUE], output_file_node.inputs[index_ob_title])
        logger.info(f"Linked {index_ob_title} output to the file node.")
    else:
        logger.error(f"Render Layers node does not contain an '{index_ob_title}' output.")


def _setup_id_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        object_index_output_configuration: NodeOutputConfiguration,
        id_mask_output_configuration: NodeOutputConfiguration,
) -> None:
    """
    Set up the biome mask output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up terrain mask output.")
    id_mask_node = node_tree.nodes.new(type=COMPOSITOR_NODE_ID_MASK)
    id_mask_node.index = 255
    id_mask_node.use_antialiasing = True

    id_mask_title = id_mask_output_configuration.title

    output_file_node.file_slots.new(id_mask_title)
    id_mask_file_slot = output_file_node.file_slots[id_mask_title]
    id_mask_file_slot.use_node_format = id_mask_output_configuration.use_node_format
    id_mask_file_slot.format.file_format = id_mask_output_configuration.file_format
    id_mask_file_slot.format.color_mode = id_mask_output_configuration.color_mode
    id_mask_file_slot.path = id_mask_output_configuration.path

    index_ob_title = object_index_output_configuration.title

    id_mask_output = render_layers.outputs.get(index_ob_title)
    if id_mask_output:
        _ = node_tree.links.new(id_mask_output, id_mask_node.inputs[ID_VALUE])
        _ = node_tree.links.new(id_mask_node.outputs[ALPHA], output_file_node.inputs[id_mask_title])
        logger.info(f"Linked {id_mask_title} output to the file node.")
    else:
        logger.error(f"Render Layers node does not contain an '{index_ob_title}' output for {id_mask_title}.")


def _setup_environment_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        environment_output_configuration: NodeOutputConfiguration
) -> None:
    """
    Set up the environment mask output.

    Args:
        node_tree (bpy.types.CompositorNodeTree): The compositor node tree.
        output_file_node (bpy.types.CompositorNodeOutputFile): The output file node.
        render_layers (bpy.types.CompositorNodeRLayers): The render layers node.
    """
    logger.debug("Setting up environment mask output.")

    environment_title = environment_output_configuration.title

    output_file_node.file_slots.new(environment_title)
    hdri_mask_file_slot = output_file_node.file_slots[environment_title]
    hdri_mask_file_slot.use_node_format = environment_output_configuration.use_node_format
    hdri_mask_file_slot.format.file_format = environment_output_configuration.file_format
    hdri_mask_file_slot.format.color_mode = environment_output_configuration.color_mode
    hdri_mask_file_slot.path = environment_output_configuration.path

    env_output = render_layers.outputs.get(ENV)
    if env_output:
        _ = node_tree.links.new(env_output, output_file_node.inputs[environment_title])
        logger.info(f"Linked {environment_title} output to the file node.")
    else:
        logger.error("Render Layers node does not contain an 'Env' output.")
