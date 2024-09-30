import bpy

from typing import List
import logging

from consts import Constants
from configs.configuration import RenderConfiguration, CameraConfiguration, RenderType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_rendering(
        render_configuration: RenderConfiguration,
        camera_configuration: CameraConfiguration,
) -> None:
    logger.info("Setting up rendering configuration...")

    scene = bpy.context.scene
    render: bpy.types.RenderSettings = scene.render

    render.engine = render_configuration.render.value
    logger.info(f"Configured render engine: {scene.render.engine}")

    _setup_camera(render, camera_configuration)
    _setup_render(render, render_configuration)

    logger.info("Rendering configuration set up.")


def _setup_camera(render: bpy.types.RenderSettings, camera_configuration: CameraConfiguration):
    """Sets up camera configuration."""
    render.resolution_x = camera_configuration.image_width
    render.resolution_y = camera_configuration.image_height


def _setup_render(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration):
    """Sets up render configuration."""
    render.resolution_percentage = 100
    render.image_settings.file_format = Constants.Render.FILE_FORMAT
    render.use_border = True
    render.use_persistent_data = True  # This helps reuse data between renders, reducing computation time
    render.threads_mode = Constants.Render.THREADS_MODE
    render.threads = Constants.Render.THREADS
    render.image_settings.compression = Constants.Render.COMPRESSION

    if render.engine == RenderType.Cycles:
        _setup_cycles(render, render_configuration)
    elif render_configuration.render == RenderType.Eevee:
        _setup_eevee()


def _setup_cycles(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration):
    """Configures Cycles rendering settings."""
    scene: bpy.types.Scene = bpy.data.scenes["Scene"]
    cycles: bpy.types.CyclesRenderSettings = scene.cycles

    cycles.camera_cull_margin = render_configuration.camera_cull_margin
    cycles.distance_cull_margin = render_configuration.distance_cull_margin
    cycles.use_camera_cull = True
    cycles.use_distance_cull = True

    cycles.feature_set = "SUPPORTED"
    cycles.device = "GPU"
    cycles.tile_size = 4096
    cycles.samples = max(1, render_configuration.n_cycles)
    cycles.use_denoising = True
    cycles.denoising_use_gpu = True

    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = Constants.Default.NOISE_THRESHOLD
    cycles.time_limit = 240

    scene.view_settings.view_transform = "Khronos PBR Neutral"

    _setup_cuda_devices(render)


def _setup_eevee():
    """Configures Eevee-specific rendering settings."""
    scene: bpy.types.Scene = bpy.data.scenes["Scene"]
    eevee: bpy.types.SceneEEVEE = scene.eevee


def _setup_cuda_devices(render: bpy.types.RenderSettings):
    """Configures CUDA devices for rendering."""
    preferences: bpy.types.AddonPreferences = bpy.context.preferences.addons[render.engine.lower()].preferences
    preferences.compute_device_type = "CUDA"

    devices: List[bpy.types.bpy_prop_collection] = preferences.get_devices() or preferences.devices
    assert devices is not None, "No CUDA devices found"

    # Disable all devices first
    for device in devices:
        device.use = False

    # Enable the specified devices
    for index in _get_gpu_indices(devices, preferences.default_device()):
        devices[index].use = True


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
