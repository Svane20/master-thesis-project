from pathlib import Path
from typing import List

import bpy
from consts import Constants
from configs.configuration import RenderConfiguration, CameraConfiguration


class RenderingConstants:
    class Engine:
        CYCLES: str = "CYCLES"
        EEVEE: str = "BLENDER_EEVEE_NEXT"
        WORKBENCH: str = "BLENDER_WORKBENCH"

    class Render:
        SOLUTION_PERCENTAGE: int = 100
        THREADS_MODE: str = "FIXED"
        THREADS: int = 54

    class ImageSettings:
        FILE_FORMAT: str = "PNG"
        COLOR_MODE: str = "RGBA"
        COMPRESSION: int = 0

    class Scene:
        NAME: str = "Scene"
        CYCLES_FEATURE_SET: str = "SUPPORTED"
        CYCLES_DEVICE: str = "GPU"
        CYCLES_TILE_SIZE: int = 4096
        CYCLES_TIME_LIMIT: int = 240
        VIEW_SETTINGS_VIEW_TRANSFORM: str = "Khronos PBR Neutral"

    class Preferences:
        COMPUTE_DEVICE_TYPE: str = "CUDA"

    class NodeTree:
        RENDER_LAYERS: str = "Render Layers"
        COMPOSITOR_NODE_OUTPUT_FILE: str = "CompositorNodeOutputFile"
        FILE_OUTPUT: str = "File Output"

        class FileSlots:
            IMAGE: str = "Image"
            OBJECT_INDEX: str = "Object Index"
            MATERIAL_INDEX: str = "Material Index"
            DEPTH: str = "Depth"
            NORMAL: str = "Normal"
            MIST: str = "Mist"


def set_noise_threshold(noise_threshold: float = Constants.Default.NOISE_THRESHOLD) -> None:
    """Sets the noise threshold for adaptive sampling in the Cycles engine."""
    cycles: bpy.types.CyclesRenderSettings = bpy.context.scene.cycles

    if noise_threshold > 0:
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = noise_threshold
    else:
        cycles.use_adaptive_sampling = False

    cycles.time_limit = RenderingConstants.Scene.CYCLES_TIME_LIMIT


def setup_rendering(
        render_configuration: RenderConfiguration,
        camera_configuration: CameraConfiguration,
        output_dir: Path = Constants.Directory.OUTPUT_DIR,
        output_name: str = None,
        render_image: bool = True,
        render_object_index: bool = True,
        render_material_index: bool = False,
        render_depth: bool = False,
        render_mist: bool = False,
        render_normal: bool = False,
        world_size: int = Constants.Default.WORLD_SIZE,
) -> None:
    """Sets up rendering configuration in Blender using the provided settings."""
    scene: bpy.types.Scene = bpy.context.scene
    render: bpy.types.RenderSettings = scene.render
    cycles: bpy.types.CyclesRenderSettings = scene.cycles

    # Render configuration
    render.engine = render_configuration.render.value
    render.resolution_x = int(camera_configuration.image_width)
    render.resolution_y = int(camera_configuration.image_height)

    # Cycles configuration
    cycles.camera_cull_margin = render_configuration.camera_cull_margin
    cycles.distance_cull_margin = render_configuration.distance_cull_margin
    cycles.use_camera_cull = True
    cycles.use_distance_cull = True

    # Setup CUDA if applicable
    _setup_cuda(render_configuration)

    # @Todo: Come back to this to understand the use-case
    # Output configuration
    # _setup_outputs(
    #     render_image=render_image,
    #     render_object_index=render_object_index,
    #     render_material_index=render_material_index,
    #     render_depth=render_depth,
    #     render_normal=render_normal,
    #     render_mist=render_mist,
    #     output_dir=output_dir,
    #     output_name=output_name,
    #     world_size=world_size,
    # )


def _setup_outputs(
        render_image: bool = True,
        render_object_index: bool = True,
        render_material_index: bool = True,
        render_mist: bool = True,
        render_depth: bool = True,
        render_normal: bool = True,
        world_size: int = 100,
        output_dir: Path = Constants.Directory.OUTPUT_DIR,
        output_name: str = None
) -> None:
    """Sets up the render outputs for the Blender scene."""
    scene: bpy.types.Scene = bpy.context.scene
    view_layer: bpy.types.ViewLayer = scene.view_layers[0]

    # Configure view layer passes
    view_layer.use_pass_object_index = render_object_index
    view_layer.use_pass_material_index = render_material_index
    view_layer.use_pass_normal = render_normal
    view_layer.use_pass_z = render_depth

    # Configure scene settings
    scene.render.use_persistent_data = True
    scene.use_nodes = True

    node_tree: bpy.types.NodeTree = scene.node_tree
    nodes: bpy.types.Nodes = node_tree.nodes
    render_layers: bpy.types.CompositorNodeRLayers = nodes.get(RenderingConstants.NodeTree.RENDER_LAYERS)

    nodes.new(type=RenderingConstants.NodeTree.COMPOSITOR_NODE_OUTPUT_FILE)
    output_file_node: bpy.types.CompositorNodeOutputFile = nodes.get(RenderingConstants.NodeTree.FILE_OUTPUT)
    if output_file_node is None:
        print("Output file node not found")
        return

    output_file_node.inputs.clear()
    output_file_node.base_path = output_dir.as_posix()

    if render_image:
        _setup_image_file_slot(node_tree, render_layers, output_file_node, output_name)


def _setup_image_file_slot(
        node_tree: bpy.types.NodeTree,
        render_layers: bpy.types.CompositorNodeRLayers,
        output_file_node: bpy.types.CompositorNodeOutputFile,
        output_name: str
) -> None:
    """Sets up the file output node for the image render pass."""
    image_name_const = RenderingConstants.NodeTree.FileSlots.IMAGE

    # Create a new file slot for the image render pass
    output_file_node.file_slots.new(image_name_const)
    image_file_slot: bpy.types.NodeOutputFileSlotFile = output_file_node.file_slots[image_name_const]

    # Configure the image file slot with custom format settings
    image_file_slot.use_node_format = False
    image_file_slot.format.file_format = RenderingConstants.ImageSettings.FILE_FORMAT
    image_file_slot.format.color_mode = RenderingConstants.ImageSettings.COLOR_MODE

    if output_name is not None:
        image_file_slot.path = output_name
    else:
        image_file_slot.path = image_name_const

    # Get the image output from the render layers
    image_output: bpy.types.NodeSocket = render_layers.outputs.get(image_name_const)
    if image_output is None:
        print(f"Render layer output '{image_name_const}' not found")
        return

    # Link the render layer image output to the file output node
    _ = node_tree.links.new(image_output, output_file_node.inputs[image_name_const])


def _setup_cuda(render_configuration: RenderConfiguration) -> None:
    """Configures CUDA settings for the rendering process."""
    scene: bpy.types.Scene = bpy.data.scenes[RenderingConstants.Scene.NAME]
    render: bpy.types.RenderSettings = scene.render

    # General Render configuration
    _configure_render_settings(render)

    if render.engine == RenderingConstants.Engine.CYCLES:
        # Cycles-specific configuration
        _configure_cycles_settings(render_configuration)
        preferences: bpy.types.CyclesPreferences = bpy.context.preferences.addons[render.engine.lower()].preferences
        _configure_cuda_devices(preferences)


def _configure_render_settings(render: bpy.types.RenderSettings) -> None:
    """Configures general render settings."""
    render.resolution_percentage = RenderingConstants.Render.SOLUTION_PERCENTAGE
    render.image_settings.file_format = RenderingConstants.ImageSettings.FILE_FORMAT
    render.use_border = True
    render.use_persistent_data = True
    render.threads_mode = RenderingConstants.Render.THREADS_MODE
    render.threads = RenderingConstants.Render.THREADS
    render.image_settings.compression = RenderingConstants.ImageSettings.COMPRESSION


def _configure_cycles_settings(render_configuration: RenderConfiguration) -> None:
    """Configures Cycles-specific settings."""
    cycles: bpy.types.CyclesRenderSettings = bpy.context.scene.cycles
    cycles.feature_set = RenderingConstants.Scene.CYCLES_FEATURE_SET
    cycles.device = RenderingConstants.Scene.CYCLES_DEVICE
    cycles.tile_size = RenderingConstants.Scene.CYCLES_TILE_SIZE
    cycles.samples = max(1, render_configuration.n_cycles)
    cycles.use_denoising = True
    cycles.denoising_use_gpu = True

    bpy.context.scene.view_settings.view_transform = RenderingConstants.Scene.VIEW_SETTINGS_VIEW_TRANSFORM


# bpy.types.CyclesPreferences is not supported as input type but is a subclass of bpy.types.AddonPreferences
def _configure_cuda_devices(preferences: bpy.types.AddonPreferences) -> None:
    """Configures CUDA devices based on the render configuration."""
    preferences.compute_device_type = RenderingConstants.Preferences.COMPUTE_DEVICE_TYPE

    devices: List[bpy.types.bpy_prop_collection] = preferences.get_devices() or preferences.devices
    assert devices is not None, "No CUDA devices found"

    # Disable all devices first
    for device in devices:
        device.use = False

    # Enable the specified devices
    for index in _get_gpu_indices(devices, preferences.default_device()):
        devices[index].use = True

    # Ensure at least the primary device is enabled
    if not any(device.use for device in devices):
        devices[0].use = True


def _get_gpu_indices(devices: List[bpy.types.bpy_prop_collection], default_device: int) -> List[int]:
    """Returns the indices of the GPU devices."""
    num_devices = len(devices)
    if num_devices == 0:
        return []

    # Primary device
    gpu_indices = [default_device]

    if num_devices > 2:
        # If there are more than 2 devices, skip the second index (CPU)
        gpu_indices.append(2)

    return gpu_indices
