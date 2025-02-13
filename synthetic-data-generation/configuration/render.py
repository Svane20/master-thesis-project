from enum import Enum
from pydantic import BaseModel

from configuration.outputs import OutputsConfiguration


class EngineType(str, Enum):
    """
    Enum representing the available rendering engines in Blender.

    Attributes:
        Cycles: Use the Cycles rendering engine.
        Eevee: Use the Eevee rendering engine.
        Workbench: Use the Workbench rendering engine.
    """
    Cycles = "CYCLES"
    Eevee = "BLENDER_EEVEE_NEXT"
    Workbench = "BLENDER_WORKBENCH"


class CyclesConfiguration(BaseModel):
    """
    Configuration for the Cycles rendering engine in Blender.

    Attributes:
        camera_cull_margin (float): The margin for camera culling.
        distance_cull_margin (float): The margin for distance culling.
        use_camera_cull (bool): Whether to use camera culling.
        use_distance_cull (bool): Whether to use distance culling.
        device (str): The rendering device (e.g., GPU).
        tile_size (int): The size of render tiles.
        samples (int): The number of samples to use for rendering.
        use_denoising (bool): Whether to use denoising.
        denoising_use_gpu (bool): Whether to use GPU for denoising.
        use_adaptive_sampling (bool): Whether to use adaptive sampling.
        adaptive_threshold (float): The threshold for adaptive sampling.
        time_limit (int): Time limit for rendering in seconds.
        view_transform (str): The view transform used in rendering.
    """
    camera_cull_margin: float
    distance_cull_margin: float
    use_camera_cull: bool
    use_distance_cull: bool
    feature_set: str
    device: str
    tile_size: int
    samples: int
    use_denoising: bool
    denoising_use_gpu: bool
    use_adaptive_sampling: bool
    adaptive_threshold: float
    time_limit: int
    view_transform: str


class PreferencesConfiguration(BaseModel):
    """
    Configuration for Blender preferences.

    Attributes:
        compute_device_type (str): The compute device type (e.g., CUDA).
    """
    compute_device_type: str


class RenderConfiguration(BaseModel):
    """
    Main configuration class for rendering settings in Blender.

    Attributes:
        engine (EngineType): The rendering engine to use.
        temp_folder (str): The temporary folder for intermediate files.
        resolution_percentage (int): The percentage of resolution for rendering.
        file_format (str): The file format for output images.
        use_border (bool): Whether to use border rendering.
        use_persistent_data (bool): Whether to use persistent data for rendering.
        threads_mode (str): The mode for using threads (e.g., FIXED).
        threads (int): The number of threads to use for rendering.
        compression (int): The compression level for output files.
        cycles_configuration (CyclesConfiguration): The configuration for the Cycles rendering engine.
        preferences_configuration (PreferencesConfiguration): The configuration for Blender preferences.
        outputs_configuration (OutputsConfiguration): The configuration for output settings.
    """
    engine: str
    temp_folder: str
    resolution_percentage: int
    file_format: str
    use_border: bool
    use_persistent_data: bool
    threads_mode: str
    threads: int
    compression: int
    cycles_configuration: CyclesConfiguration
    preferences_configuration: PreferencesConfiguration
    outputs_configuration: OutputsConfiguration
