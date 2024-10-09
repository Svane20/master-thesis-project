import bpy

from typing import List

from configuration.configuration import RenderConfiguration, CameraConfiguration, EngineType
from custom_logging.custom_logger import setup_logger
from engine.rendering_outputs import setup_outputs
from utils.utils import get_temporary_file_path

logger = setup_logger(__name__)

SCENE = "Scene"
CYCLES_SAMPLES = 1


def setup_rendering(
        render_configuration: RenderConfiguration,
        camera_configuration: CameraConfiguration,
) -> None:
    """
    Sets up rendering configuration.

    Args:
        render_configuration: The render configuration.
        camera_configuration: The camera configuration.

    Raises:
        Exception: If the render engine is not supported.
    """
    logger.info("Setting up rendering configuration...")

    scene = bpy.context.scene
    render: bpy.types.RenderSettings = scene.render

    render.engine = render_configuration.engine.value
    logger.info(f"Configured render engine: {scene.render.engine}")

    render.filepath = get_temporary_file_path(render_configuration)
    logger.info(f"Render output path: {render.filepath}")

    _setup_camera(render, camera_configuration)
    _setup_render(render, render_configuration)

    setup_outputs(scene, render_configuration)

    logger.info("Rendering configuration set up.")


def _setup_camera(render: bpy.types.RenderSettings, camera_configuration: CameraConfiguration) -> None:
    """
    Sets up camera configuration.

    Args:
        render: The render settings.
        camera_configuration: The camera configuration.
    """
    render.resolution_x = camera_configuration.image_width
    render.resolution_y = camera_configuration.image_height


def _setup_render(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Sets up render configuration.

    Args:
        render: The render settings.
        render_configuration: The render configuration
    """
    logger.info("Setting up render configuration...")

    render.resolution_percentage = render_configuration.resolution_percentage
    render.image_settings.file_format = render_configuration.file_format
    render.use_border = render_configuration.use_border
    render.use_persistent_data = render_configuration.use_persistent_data
    render.threads_mode = render_configuration.threads_mode
    render.threads = render_configuration.threads
    render.image_settings.compression = render_configuration.compression

    if render_configuration.engine == EngineType.Cycles:
        _setup_cycles(render, render_configuration)

    logger.info("Render configuration set up.")


def _setup_cycles(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Configures Cycles rendering settings.

    Args:
        render: The render settings.
        render_configuration: The render configuration.
    """
    logger.info("Setting up Cycles rendering configuration...")

    cycles_configuration = render_configuration.cycles_configuration

    scene: bpy.types.Scene = bpy.data.scenes[SCENE]
    cycles: bpy.types.CyclesRenderSettings = scene.cycles

    cycles.camera_cull_margin = cycles_configuration.camera_cull_margin
    cycles.distance_cull_margin = cycles_configuration.distance_cull_margin
    cycles.use_camera_cull = cycles_configuration.use_camera_cull
    cycles.use_distance_cull = cycles_configuration.use_distance_cull

    cycles.feature_set = cycles_configuration.feature_set
    cycles.device = cycles_configuration.device
    cycles.tile_size = cycles_configuration.tile_size
    cycles.samples = max(CYCLES_SAMPLES, cycles_configuration.samples)
    cycles.use_denoising = cycles_configuration.use_denoising
    cycles.denoising_use_gpu = cycles_configuration.denoising_use_gpu

    cycles.use_adaptive_sampling = cycles_configuration.use_adaptive_sampling
    cycles.adaptive_threshold = cycles_configuration.adaptive_threshold
    cycles.time_limit = cycles_configuration.time_limit

    scene.view_settings.view_transform = cycles_configuration.view_transform

    logger.info("Cycles rendering configuration set up.")

    _setup_cuda_devices(render, render_configuration)


def _setup_cuda_devices(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Configures CUDA devices for rendering.

    Args:
        render: The render settings.
        render_configuration: The render configuration.
    """
    logger.info("Setting up CUDA devices for rendering...")

    preferences_configuration = render_configuration.preferences_configuration

    preferences: bpy.types.AddonPreferences = bpy.context.preferences.addons[render.engine.lower()].preferences
    preferences.compute_device_type = preferences_configuration.compute_device_type

    devices: List[bpy.types.bpy_prop_collection] = preferences.get_devices() or preferences.devices
    assert devices is not None, "No CUDA devices found"

    # Disable all devices first
    for device in devices:
        device.use = False

    # Enable the specified devices
    for index in _get_gpu_indices(devices, preferences.default_device()):
        devices[index].use = True

    enabled_devices = [device.name for device in devices if device.use]
    logger.info(f"Enabled CUDA devices: {enabled_devices}")


def _get_gpu_indices(devices: List[bpy.types.bpy_prop_collection], default_device: int) -> List[int]:
    """
    Returns the indices of the GPU devices.

    Args:
        devices: The list of devices.
        default_device: The default device index.

    Returns:
        The list of GPU indices.
    """
    logger.info("Getting GPU indices...")

    num_devices = len(devices)
    if num_devices == 0:
        return []

    # Primary device
    gpu_indices = [default_device]

    if num_devices > 2:
        # If there are more than 2 devices, skip the second index (CPU)
        gpu_indices.append(2)

    logger.info(f"GPU indices: {gpu_indices}")

    return gpu_indices
