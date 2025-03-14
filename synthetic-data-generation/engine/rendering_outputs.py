import bpy
import logging

from configuration.outputs import OutputsConfiguration, NodeOutputConfiguration
from configuration.render import EngineType


class Constants:
    RENDER_LAYERS: str = "Render Layers"
    FILE_OUTPUT: str = "File Output"
    COMPOSITOR_NODE_MAP_RANGE: str = "CompositorNodeMapRange"
    COMPOSITOR_NODE_ID_MASK: str = "CompositorNodeIDMask"
    COMPOSITOR_NODE_OUTPUT_FILE: str = "CompositorNodeOutputFile"
    COMPOSITOR_NODE_RGB_TO_BW: str = "CompositorNodeRGBToBW"
    COMPOSITOR_NODE_MAP_VALUE: str = "CompositorNodeMapValue"
    COMPOSITOR_NODE_VAL_TO_RGB: str = "CompositorNodeValToRGB"
    COMPOSITOR_NODE_COMB_RGBA: str = "CompositorNodeCombRGBA"
    COMPOSITOR_NODE_MATH: str = "CompositorNodeMath"
    FROM_MIN: str = "From Min"
    FROM_MAX: str = "From Max"
    TO_MIN: str = "To Min"
    TO_MAX: str = "To Max"
    VALUE: str = "Value"
    VAL: str = "Val"
    ID_VALUE: str = "ID value"
    ALPHA: str = "Alpha"
    ENV: str = "Env"
    IMAGE: str = "Image"
    LINEAR: str = "LINEAR"
    FAC: str = "Fac"
    SUBTRACT: str = "SUBTRACT"
    R: str = "R"
    G: str = "G"
    B: str = "B"
    A: str = "A"


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
    """
    logging.info("Setting up rendering outputs...")

    view_layer: bpy.types.ViewLayer = scene.view_layers[0]

    # Configure rendering passes based on the configuration
    view_layer.use_pass_object_index = render_object_index and engine_type == EngineType.Cycles.value
    view_layer.use_pass_environment = render_environment and engine_type == EngineType.Cycles.value
    view_layer.use_pass_material_index = False
    view_layer.use_pass_normal = False
    view_layer.use_pass_z = False

    logging.debug(f"Passes configured", extra={
        "Grass Mask": view_layer.use_pass_object_index,
        "Sky Mask": view_layer.use_pass_environment
    })

    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.CompositorNodeTree = scene.node_tree
    render_layers: bpy.types.CompositorNodeRLayers = node_tree.nodes.get(Constants.RENDER_LAYERS)

    output_file_node: bpy.types.CompositorNodeOutputFile = (
            node_tree.nodes.get(Constants.FILE_OUTPUT)
            or node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_OUTPUT_FILE)
    )
    output_file_node.inputs.clear()
    output_file_node.base_path = output_path
    logging.debug(f"Output file node set to: {output_file_node.base_path}")

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

    logging.info("Render outputs configured.", extra={
        "Render Image": render_image,
        "Render Grass Mask": view_layer.use_pass_object_index,
        "Render Sky Mask": view_layer.use_pass_environment
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
    logging.debug("Setting up image output.")
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
        logging.debug(f"Linked {image_title} output to the file node.")
    else:
        logging.error(f"Render Layers node does not contain an '{image_title}' output.")


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
    logging.debug("Setting up object index output.")
    map_range_node_indexOB = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_MAP_RANGE)
    map_range_node_indexOB.inputs[Constants.FROM_MIN].default_value = 0
    map_range_node_indexOB.inputs[Constants.FROM_MAX].default_value = 255
    map_range_node_indexOB.inputs[Constants.TO_MIN].default_value = 0
    map_range_node_indexOB.inputs[Constants.TO_MAX].default_value = 1

    index_ob_title = object_index_output_configuration.title

    output_file_node.file_slots.new(index_ob_title)
    index_ob_file_slot = output_file_node.file_slots[index_ob_title]
    index_ob_file_slot.use_node_format = object_index_output_configuration.use_node_format
    index_ob_file_slot.format.file_format = object_index_output_configuration.file_format
    index_ob_file_slot.format.color_mode = object_index_output_configuration.color_mode
    index_ob_file_slot.path = object_index_output_configuration.path

    index_ob_output = render_layers.outputs.get(index_ob_title)
    if index_ob_output:
        _ = node_tree.links.new(index_ob_output, map_range_node_indexOB.inputs[Constants.VALUE])
        _ = node_tree.links.new(map_range_node_indexOB.outputs[Constants.VALUE],
                                output_file_node.inputs[index_ob_title])
        logging.debug(f"Linked {index_ob_title} output to the file node.")
    else:
        logging.error(f"Render Layers node does not contain an '{index_ob_title}' output.")


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
    logging.debug("Setting up terrain mask output.")
    id_mask_node = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_ID_MASK)
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
        _ = node_tree.links.new(id_mask_output, id_mask_node.inputs[Constants.ID_VALUE])
        _ = node_tree.links.new(id_mask_node.outputs[Constants.ALPHA], output_file_node.inputs[id_mask_title])
        logging.debug(f"Linked {id_mask_title} output to the file node.")
    else:
        logging.error(f"Render Layers node does not contain an '{index_ob_title}' output for {id_mask_title}.")


def _setup_environment_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        environment_output_configuration: NodeOutputConfiguration
) -> None:
    """
    Generate a strictly binary (0 or 255) sky/HDRI mask without using film_transparent.
    The mask is built from the environment (Env) pass:
        - It converts the Env pass to grayscale,
        - Applies a threshold to separate sky from any non-sky (if any),
        - Then maps the result to 0 or 255.
    """
    logging.debug("Setting up sky mask output using the Env pass.")

    environment_title = environment_output_configuration.title

    # Create and configure the file output slot for the sky mask.
    output_file_node.file_slots.new(environment_title)
    hdri_mask_file_slot = output_file_node.file_slots[environment_title]
    hdri_mask_file_slot.use_node_format = environment_output_configuration.use_node_format
    hdri_mask_file_slot.format.file_format = environment_output_configuration.file_format
    hdri_mask_file_slot.format.color_mode = environment_output_configuration.color_mode
    hdri_mask_file_slot.path = environment_output_configuration.path

    # Get the environment (sky) pass from the render layers.
    env_output = render_layers.outputs.get(Constants.ENV)
    if not env_output:
        logging.error("Render Layers node does not contain an 'Env' output.")
        return

    # Convert the Env pass to grayscale.
    rgb_to_bw = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_RGB_TO_BW)
    node_tree.links.new(env_output, rgb_to_bw.inputs[0])

    # Map the grayscale output from [0,1] to [0,255] (preserving soft transitions).
    map_range_node = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_MAP_RANGE)
    map_range_node.inputs[Constants.FROM_MIN].default_value = 0.0
    map_range_node.inputs[Constants.FROM_MAX].default_value = 1.0
    map_range_node.inputs[Constants.TO_MIN].default_value = 0.0
    map_range_node.inputs[Constants.TO_MAX].default_value = 255.0
    node_tree.links.new(rgb_to_bw.outputs[0], map_range_node.inputs[Constants.VALUE])

    # Connect the mapped output to the file output node.
    node_tree.links.new(map_range_node.outputs[Constants.VALUE], output_file_node.inputs[environment_title])

    logging.debug(f"Linked {environment_title} output as a sky mask using the Env pass, thresholding, and map range.")
