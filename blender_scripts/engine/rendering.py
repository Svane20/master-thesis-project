import bpy

from typing import List

from configuration.consts import Constants
from configuration.configuration import RenderConfiguration, CameraConfiguration, RenderType
from custom_logging.custom_logger import setup_logger
from engine.rendering_outputs import setup_outputs
from utils.utils import get_temporary_file_path

logger = setup_logger(__name__)


class RenderingConstants:
    class Default:
        SCENE = "Scene"
        VIEW_TRANSFORM = "Khronos PBR Neutral"

    class Render:
        RESOLUTION_PERCENTAGE = 100

    class Cycles:
        FEATURE_SET = "SUPPORTED"
        DEVICE = "GPU"
        TILE_SIZE = 4096
        SAMPLES = 1
        ADAPTIVE_THRESHOLD = 0.0
        TIME_LIMIT = 240

    class Preferences:
        COMPUTE_DEVICE_TYPE = "CUDA"


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

    render.engine = render_configuration.render.value
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

    render.resolution_percentage = RenderingConstants.Render.RESOLUTION_PERCENTAGE
    render.image_settings.file_format = Constants.Render.FILE_FORMAT
    render.use_border = True
    render.use_persistent_data = True  # This helps reuse data between renders, reducing computation time
    render.threads_mode = Constants.Render.THREADS_MODE
    render.threads = Constants.Render.THREADS
    render.image_settings.compression = Constants.Render.COMPRESSION

    _setup_cycles(render, render_configuration) if render_configuration.render == RenderType.Cycles else _setup_eevee()

    logger.info("Render configuration set up.")


def _setup_cycles(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Configures Cycles rendering settings.

    Args:
        render: The render settings.
        render_configuration: The render configuration.
    """
    logger.info("Setting up Cycles rendering configuration...")

    scene: bpy.types.Scene = bpy.data.scenes[RenderingConstants.Default.SCENE]
    cycles: bpy.types.CyclesRenderSettings = scene.cycles

    cycles.camera_cull_margin = render_configuration.camera_cull_margin
    cycles.distance_cull_margin = render_configuration.distance_cull_margin
    cycles.use_camera_cull = True
    cycles.use_distance_cull = True

    cycles.feature_set = RenderingConstants.Cycles.FEATURE_SET
    cycles.device = RenderingConstants.Cycles.DEVICE
    cycles.tile_size = RenderingConstants.Cycles.TILE_SIZE
    cycles.samples = max(RenderingConstants.Cycles.SAMPLES, render_configuration.n_cycles)
    cycles.use_denoising = True
    cycles.denoising_use_gpu = True

    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = Constants.Default.NOISE_THRESHOLD
    cycles.time_limit = RenderingConstants.Cycles.TIME_LIMIT

    scene.view_settings.view_transform = RenderingConstants.Default.VIEW_TRANSFORM

    logger.info("Cycles rendering configuration set up.")

    _setup_cuda_devices(render)


def _setup_eevee() -> None:
    """Configures Eevee-specific rendering settings."""
    logger.info("Setting up Eevee rendering configuration...")

    scene: bpy.types.Scene = bpy.data.scenes[RenderingConstants.Default.SCENE]
    eevee: bpy.types.SceneEEVEE = scene.eevee

    logger.info("Eevee rendering configuration set up.")


def _setup_cuda_devices(render: bpy.types.RenderSettings) -> None:
    """
    Configures CUDA devices for rendering.

    Args:
        render: The render settings.
    """
    logger.info("Setting up CUDA devices for rendering...")

    preferences: bpy.types.AddonPreferences = bpy.context.preferences.addons[render.engine.lower()].preferences
    preferences.compute_device_type = RenderingConstants.Preferences.COMPUTE_DEVICE_TYPE

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
