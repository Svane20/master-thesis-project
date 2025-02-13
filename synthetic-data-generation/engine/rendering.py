import bpy
from typing import List
import logging

from configuration.camera import CameraConfiguration
from configuration.render import RenderConfiguration, EngineType
from engine.rendering_outputs import setup_outputs
from utils.utils import get_temporary_file_path


class Constants:
    SCENE: str = "Scene"
    CYCLES_SAMPLES: str = 1


def setup_rendering(
        render_configuration: RenderConfiguration,
        camera_configuration: CameraConfiguration,
) -> None:
    """
    Sets up the rendering configuration for the scene.

    Args:
        render_configuration (RenderConfiguration): The render configuration.
        camera_configuration (CameraConfiguration): The camera configuration.

    Raises:
        Exception: If the render engine is not supported.
    """
    logging.info("Starting rendering setup...")

    scene = bpy.context.scene
    render: bpy.types.RenderSettings = scene.render

    render.engine = render_configuration.engine
    logging.info(f"Render engine set to: {render.engine}")

    render.filepath = get_temporary_file_path(render_configuration)
    logging.info(f"Render output path: {render.filepath}")

    _setup_camera(render, camera_configuration)
    _setup_render(render, render_configuration)

    outputs_configuration = render_configuration.outputs_configuration

    setup_outputs(
        scene=scene,
        engine_type=render_configuration.engine,
        outputs_configuration=outputs_configuration,
        render_image=outputs_configuration.render_image,
        render_object_index=outputs_configuration.render_object_index,
        render_environment=outputs_configuration.render_environment,
        output_path=outputs_configuration.output_path
    )

    logging.info("Rendering configuration complete.")


def _setup_camera(render: bpy.types.RenderSettings, camera_configuration: CameraConfiguration) -> None:
    """
    Configures the camera resolution.

    Args:
        render (bpy.types.RenderSettings): The render settings.
        camera_configuration (CameraConfiguration): The camera configuration.
    """
    logging.debug("Setting up camera resolution.")
    render.resolution_x = camera_configuration.image_width
    render.resolution_y = camera_configuration.image_height
    logging.info(f"Camera resolution set to: {render.resolution_x}x{render.resolution_y}")


def _setup_render(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Configures the render settings based on the provided render configuration.

    Args:
        render (bpy.types.RenderSettings): The render settings.
        render_configuration (RenderConfiguration): The render configuration.
    """
    logging.info("Configuring general render settings...")

    render.resolution_percentage = render_configuration.resolution_percentage
    render.image_settings.file_format = render_configuration.file_format
    render.use_border = render_configuration.use_border
    render.use_persistent_data = render_configuration.use_persistent_data
    render.threads_mode = render_configuration.threads_mode
    render.threads = render_configuration.threads
    render.image_settings.compression = render_configuration.compression

    logging.info("General render settings configured.")

    if render_configuration.engine == EngineType.Cycles:
        _setup_cycles(render, render_configuration)


def _setup_cycles(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Configures Cycles-specific rendering settings.

    Args:
        render (bpy.types.RenderSettings): The render settings.
        render_configuration (RenderConfiguration): The render configuration for Cycles.
    """
    logging.info("Setting up Cycles rendering configuration...")

    cycles_configuration = render_configuration.cycles_configuration

    scene: bpy.types.Scene = bpy.data.scenes[Constants.SCENE]
    cycles: bpy.types.CyclesRenderSettings = scene.cycles

    cycles.camera_cull_margin = cycles_configuration.camera_cull_margin
    cycles.distance_cull_margin = cycles_configuration.distance_cull_margin
    cycles.use_camera_cull = cycles_configuration.use_camera_cull
    cycles.use_distance_cull = cycles_configuration.use_distance_cull

    cycles.feature_set = cycles_configuration.feature_set
    cycles.device = cycles_configuration.device
    cycles.tile_size = cycles_configuration.tile_size
    cycles.samples = max(Constants.CYCLES_SAMPLES, cycles_configuration.samples)
    cycles.use_denoising = cycles_configuration.use_denoising
    cycles.denoising_use_gpu = cycles_configuration.denoising_use_gpu

    cycles.use_adaptive_sampling = cycles_configuration.use_adaptive_sampling
    cycles.adaptive_threshold = cycles_configuration.adaptive_threshold
    cycles.time_limit = cycles_configuration.time_limit

    scene.view_settings.view_transform = cycles_configuration.view_transform

    logging.info("Cycles rendering configuration complete.")
    _setup_cuda_devices(render, render_configuration)


def _setup_cuda_devices(render: bpy.types.RenderSettings, render_configuration: RenderConfiguration) -> None:
    """
    Configures CUDA devices for GPU rendering.

    Args:
        render (bpy.types.RenderSettings): The render settings.
        render_configuration (RenderConfiguration): The render configuration.

    Raises:
        RuntimeError: If no CUDA devices are found.
    """
    logging.info("Setting up CUDA devices for rendering...")

    preferences_configuration = render_configuration.preferences_configuration

    preferences: bpy.types.AddonPreferences = bpy.context.preferences.addons[render.engine.lower()].preferences
    preferences.compute_device_type = preferences_configuration.compute_device_type

    devices: List[bpy.types.bpy_prop_collection] = preferences.get_devices() or preferences.devices
    if devices is None:
        logging.error("No CUDA devices found.")
        raise RuntimeError("No CUDA devices found")

    # Disable all devices first
    for device in devices:
        device.use = False

    # Enable the specified devices
    for index in _get_gpu_indices(devices, 0):
        devices[index].use = True

    enabled_devices = [device.name for device in devices if device.use]
    logging.info(f"Enabled CUDA devices: {enabled_devices}")


def _get_gpu_indices(devices: List[bpy.types.bpy_prop_collection], default_device: int) -> List[int]:
    """
    Returns the indices of the GPU devices.

    Args:
        devices (List[bpy.types.bpy_prop_collection]): The list of devices.
        default_device (int): The default device index.

    Returns:
        List[int]: The list of GPU indices.
    """
    logging.info("Getting GPU indices...")

    num_devices = len(devices)
    if num_devices == 0:
        return []

    # Primary device
    gpu_indices = [default_device]

    if num_devices > 2:
        # If there are more than 2 devices, skip the second index (CPU)
        gpu_indices.append(2)

    logging.info(f"GPU indices selected: {gpu_indices}")
    return gpu_indices
