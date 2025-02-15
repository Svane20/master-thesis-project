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

    Args:
        scene (bpy.context.scene): The Blender scene object.
        engine_type (EngineType): The render engine type.
        outputs_configuration (OutputsConfiguration): The outputs configuration settings.
        render_image (bool, optional): Whether to render the image. Defaults to True.
        render_object_index (bool, optional): Whether to render the object index. Defaults to True.
        render_environment (bool, optional): Whether to render the environment. Defaults to True.
        output_path (Path, optional): The output path for the rendered files. Defaults to OUTPUT_DIRECTORY.
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
        "Object Index": view_layer.use_pass_object_index,
        "Environment": view_layer.use_pass_environment
    })

    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.CompositorNodeTree = scene.node_tree
    render_layers: bpy.types.CompositorNodeRLayers = node_tree.nodes.get(Constants.RENDER_LAYERS)

    output_file_node: bpy.types.CompositorNodeOutputFile = node_tree.nodes.get(
        Constants.FILE_OUTPUT) or node_tree.nodes.new(
        type=Constants.COMPOSITOR_NODE_OUTPUT_FILE)
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
        logging.info(f"Linked {image_title} output to the file node.")
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
        logging.info(f"Linked {index_ob_title} output to the file node.")
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
        logging.info(f"Linked {id_mask_title} output to the file node.")
    else:
        logging.error(f"Render Layers node does not contain an '{index_ob_title}' output for {id_mask_title}.")


def _setup_environment_mask_output(
        node_tree: bpy.types.CompositorNodeTree,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        render_layers: bpy.types.CompositorNodeRLayers,
        environment_output_configuration: NodeOutputConfiguration
) -> None:
    """
    Set up the environment mask output to generate a continuous (soft) alpha mask for the sky/HDri.

    The resulting RGBA image has:
      - White RGB channels (so the image appears white),
      - An alpha channel that continuously varies from 0 (foreground/geometry) to 1 (sky/HDri),
        with intermediate values for soft transitions.
    """
    logging.debug("Setting up continuous environment alpha mask output.")

    # Create and configure a new file slot for the environment output.
    output_file_node.file_slots.new(environment_output_configuration.title)
    file_slot = output_file_node.file_slots[environment_output_configuration.title]
    file_slot.use_node_format = environment_output_configuration.use_node_format
    file_slot.format.file_format = environment_output_configuration.file_format
    file_slot.format.color_mode = environment_output_configuration.color_mode
    file_slot.path = environment_output_configuration.path

    # Get the ENV output from the Render Layers node.
    env_output = render_layers.outputs.get(Constants.ENV)
    if not env_output:
        logging.error("Render Layers node does not contain an 'Env' output.")
        return

    # Convert the ENV (RGB) pass to grayscale.
    env_to_gray = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_RGB_TO_BW)
    env_to_gray.label = "Env to Gray"
    node_tree.links.new(env_output, env_to_gray.inputs[Constants.IMAGE])

    # Use a Map Value node to stretch the grayscale range into [0,1].
    map_value = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_MAP_VALUE)
    map_value.label = "Map Value for ENV"
    map_value.offset = [0.0]  # No offset.
    map_value.size = [2.0]
    map_value.use_min = True
    map_value.min = [0.0]
    map_value.use_max = True
    map_value.max = [1.0]
    node_tree.links.new(env_to_gray.outputs[Constants.VAL], map_value.inputs[Constants.VALUE])

    # Add a Math node to subtract a small offset to reduce minor brightness spikes.
    subtract_node = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_MATH)
    subtract_node.operation = Constants.SUBTRACT
    subtract_node.inputs[1].default_value = 0.20
    subtract_node.use_clamp = True
    node_tree.links.new(map_value.outputs[Constants.VALUE], subtract_node.inputs[0])

    # Create a ColorRamp node with LINEAR interpolation to produce a continuous range.
    color_ramp = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_VAL_TO_RGB)
    color_ramp.label = "Continuous Alpha Mask Ramp"
    color_ramp.color_ramp.interpolation = Constants.LINEAR

    # Set up the stops:
    # At 0.0, output black (alpha 0).
    element0 = color_ramp.color_ramp.elements[0]
    element0.position = 0.0
    element0.color = (0, 0, 0, 1)

    # Add a stop at 1.0 for white (alpha 1).
    element_high = color_ramp.color_ramp.elements.new(1.0)
    element_high.color = (1, 1, 1, 1)

    # Link the adjusted (subtracted) grayscale values into the ColorRamp.
    node_tree.links.new(subtract_node.outputs[Constants.VALUE], color_ramp.inputs[Constants.FAC])

    # Combine the output into an RGBA image:
    # Force the RGB channels to white while using the ColorRamp output as the alpha.
    combine_rgba = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_COMB_RGBA)
    combine_rgba.label = "Combine RGBA for Continuous Mask"
    combine_rgba.inputs[Constants.R].default_value = 1.0
    combine_rgba.inputs[Constants.G].default_value = 1.0
    combine_rgba.inputs[Constants.B].default_value = 1.0
    node_tree.links.new(color_ramp.outputs[Constants.IMAGE], combine_rgba.inputs[Constants.A])

    # Link the combined RGBA image to the file output node input corresponding to this slot.
    node_tree.links.new(combine_rgba.outputs[Constants.IMAGE],
                        output_file_node.inputs[environment_output_configuration.title])

    logging.info(f"Continuous alpha mask output set up at '{environment_output_configuration.title}'.")

# def _setup_environment_mask_output(
#         node_tree: bpy.types.CompositorNodeTree,
#         output_file_node: bpy.types.CompositorNodeOutputFile,
#         render_layers: bpy.types.CompositorNodeRLayers,
#         environment_output_configuration: NodeOutputConfiguration
# ) -> None:
#     """
#     Set up the environment mask output to generate a continuous (soft) alpha mask for the sky/HDri.
#
#     The resulting RGBA image has:
#       - White RGB channels (so the image appears white),
#       - An alpha channel that continuously varies from 0 (foreground/geometry) to 1 (sky/HDri),
#         with intermediate values for soft transitions.
#     """
#     logging.debug("Setting up continuous environment alpha mask output.")
#
#     # Create and configure a new file slot for the environment output.
#     output_file_node.file_slots.new(environment_output_configuration.title)
#     file_slot = output_file_node.file_slots[environment_output_configuration.title]
#     file_slot.use_node_format = environment_output_configuration.use_node_format
#     file_slot.format.file_format = environment_output_configuration.file_format
#     file_slot.format.color_mode = environment_output_configuration.color_mode
#     file_slot.path = environment_output_configuration.path
#
#     # Get the ENV output from the Render Layers node.
#     env_output = render_layers.outputs.get(Constants.ENV)
#     if not env_output:
#         logging.error("Render Layers node does not contain an 'Env' output.")
#         return
#
#     # Convert the ENV (RGB) pass to grayscale.
#     env_to_gray = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_RGB_TO_BW)
#     env_to_gray.label = "Env to Gray"
#     node_tree.links.new(env_output, env_to_gray.inputs[Constants.IMAGE])
#
#     # Use a Map Value node to stretch the grayscale range into [0,1].
#     map_value = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_MAP_VALUE)
#     map_value.label = "Map Value for ENV"
#     map_value.offset = [0.0]  # No offset.
#     map_value.size = [3.0]  # Multiply values by 5 (tweak as needed).
#     map_value.use_min = True
#     map_value.min = [0.0]
#     map_value.use_max = True
#     map_value.max = [1.0]
#     node_tree.links.new(env_to_gray.outputs[Constants.VAL], map_value.inputs[Constants.VALUE])
#
#     # Create a ColorRamp node with LINEAR interpolation to produce a continuous range.
#     color_ramp = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_VAL_TO_RGB)
#     color_ramp.label = "Continuous Alpha Mask Ramp"
#     color_ramp.color_ramp.interpolation = Constants.LINEAR
#
#     # Set up the stops:
#     # At 0.0, output black (alpha 0).
#     element0 = color_ramp.color_ramp.elements[0]
#     element0.position = 0.0
#     element0.color = (0, 0, 0, 1)
#
#     # Add a stop at 1.0 for white (alpha 1).
#     element_high = color_ramp.color_ramp.elements.new(0.8)
#     element_high.color = (1, 1, 1, 1)
#
#     # Link the scaled grayscale values into the ColorRamp.
#     node_tree.links.new(map_value.outputs[Constants.VALUE], color_ramp.inputs[Constants.FAC])
#
#     # Combine the output into an RGBA image:
#     # Force the RGB channels to white while using the ColorRamp output as the alpha.
#     combine_rgba = node_tree.nodes.new(type=Constants.COMPOSITOR_NODE_COMB_RGBA)
#     combine_rgba.label = "Combine RGBA for Continuous Mask"
#     combine_rgba.inputs[Constants.R].default_value = 1.0
#     combine_rgba.inputs[Constants.G].default_value = 1.0
#     combine_rgba.inputs[Constants.B].default_value = 1.0
#     node_tree.links.new(color_ramp.outputs[Constants.IMAGE], combine_rgba.inputs[Constants.A])
#
#     # Link the combined RGBA image to the file output node input corresponding to this slot.
#     node_tree.links.new(combine_rgba.outputs[Constants.IMAGE],
#                         output_file_node.inputs[environment_output_configuration.title])
#
#     logging.info(f"Continuous alpha mask output set up at '{environment_output_configuration.title}'.")
