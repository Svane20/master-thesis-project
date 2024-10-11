from pathlib import Path
from pydantic import BaseModel
from typing import Union, Dict
from enum import Enum
import json
from constants.defaults import IMAGE_WIDTH, IMAGE_HEIGHT, WORLD_SIZE, IMAGE_SIZE
from constants.directories import TEMP_DIRECTORY, CONFIG_PATH, OUTPUT_DIRECTORY
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


class EngineType(str, Enum):
    Cycles = "CYCLES"
    Eevee = "BLENDER_EEVEE_NEXT"
    Workbench = "BLENDER_WORKBENCH"


class PreferencesConfiguration(BaseModel):
    compute_device_type: str = "CUDA"


class ImageOutputConfiguration(BaseModel):
    title: str = "Image"
    use_node_format: bool = False  # Custom format
    file_format: str = "PNG"
    color_mode: str = "RGBA"
    path: str = "Image"


class ObjectIndexOutputConfiguration(BaseModel):
    title: str = "IndexOB"
    use_node_format: bool = False  # Custom format
    file_format: str = "PNG"
    color_mode: str = "BW"
    path: str = "IndexOB"


class IDMaskOutputConfiguration(BaseModel):
    title: str = "BiomeMask"
    use_node_format: bool = False  # Custom format
    file_format: str = "PNG"
    color_mode: str = "BW"
    path: str = "BiomeMask"


class EnvironmentOutputConfiguration(BaseModel):
    title: str = "HDRIMask"
    use_node_format: bool = False  # Custom format
    file_format: str = "PNG"
    color_mode: str = "BW"
    path: str = "HDRIMask"


class OutputsConfiguration(BaseModel):
    render_image: bool = True
    render_object_index: bool = True
    render_environment: bool = True
    output_path: str = OUTPUT_DIRECTORY.as_posix()

    image_output_configuration: ImageOutputConfiguration = ImageOutputConfiguration()
    object_index_output_configuration: ObjectIndexOutputConfiguration = ObjectIndexOutputConfiguration()
    id_mask_output_configuration: IDMaskOutputConfiguration = IDMaskOutputConfiguration()
    environment_output_configuration: EnvironmentOutputConfiguration = EnvironmentOutputConfiguration()


class CyclesConfiguration(BaseModel):
    camera_cull_margin: float = 1.0
    distance_cull_margin: float = 200.0
    use_camera_cull: bool = True
    use_distance_cull: bool = True
    feature_set: str = "SUPPORTED"
    device: str = "GPU"
    tile_size: int = 4096
    samples: int = 1  # Set to a lower value for development, increase for production
    use_denoising: bool = True
    denoising_use_gpu: bool = True
    use_adaptive_sampling: bool = True
    adaptive_threshold: float = 0.01
    time_limit: int = 240
    view_transform: str = "Khronos PBR Neutral"


class RenderConfiguration(BaseModel):
    engine: EngineType = EngineType.Cycles
    temp_folder: str = TEMP_DIRECTORY.as_posix()
    resolution_percentage: int = 100
    file_format: str = "PNG"
    use_border: bool = True
    use_persistent_data: bool = True
    threads_mode: str = "FIXED"
    threads: int = 54
    compression: int = 0
    cycles_configuration: CyclesConfiguration = CyclesConfiguration()
    preferences_configuration: PreferencesConfiguration = PreferencesConfiguration()
    outputs_configuration: OutputsConfiguration = OutputsConfiguration()


class CameraConfiguration(BaseModel):
    image_width: int = IMAGE_WIDTH
    image_height: int = IMAGE_HEIGHT
    camera_fov_mu_deg: float = 60.0
    camera_fov_std_deg: float = 10.0


class TerrainConfiguration(BaseModel):
    world_size: float = WORLD_SIZE
    image_size: int = IMAGE_SIZE
    prob_of_trees: float = 0.25


class SunConfiguration(BaseModel):
    size: dict[str, int] = {"min": 1, "max": 3}
    elevation: dict[str, int] = {"min": 45, "max": 90}
    rotation: dict[str, int] = {"min": 0, "max": 360}
    intensity: dict[str, float] = {"min": 0.4, "max": 0.8}


class HDRIConfiguration(BaseModel):
    temperature: dict[str, int] = {"min": 5000, "max": 6500}
    strength: dict[str, float] = {"min": 0.6, "max": 1.0}
    density: dict[str, int] = {"min": 0, "max": 2}

    sun_configuration: SunConfiguration = SunConfiguration()


class Configuration(BaseModel):
    render_configuration: RenderConfiguration = RenderConfiguration()
    camera_configuration: CameraConfiguration = CameraConfiguration()
    terrain_configuration: TerrainConfiguration = TerrainConfiguration()
    hdri_configuration: HDRIConfiguration = HDRIConfiguration()


def load_configuration(path: Path = CONFIG_PATH) -> Union[Dict[str, Union[str, int, float, bool, dict]], None]:
    """
    Load settings from a JSON file.

    Args:
        path (Path): The path to the JSON configuration file to be loaded.

    Returns:
        Union[Dict[str, Union[str, int, float, bool, dict]], None]:
        The configuration data loaded from the file as a dictionary.
        Returns `None` if loading fails due to any error.

    Raises:
        Exception: If the configuration could not be loaded.
    """
    try:
        with path.open('r') as f:
            configuration = json.load(f)
        logger.info(f"Configuration successfully loaded from {path}")
        return configuration
    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}")
        return None


def save_configuration(
        configuration: Dict[str, Union[str, int, float, bool, dict]],
        path: Path = CONFIG_PATH
) -> Dict[str, Union[str, int, float, bool, dict]]:
    """
    Save settings to a JSON file.

    Args:
        configuration (Dict[str, Union[str, int, float, bool, dict]]):
            The configuration data to be saved, structured as a dictionary.
        path (Path): The path to the JSON file where the configuration should be saved.

    Returns:
        Dict[str, Union[str, int, float, bool, dict]]: The configuration data that was saved.

    Raises:
        Exception: If the configuration could not be saved.
    """
    try:
        with path.open('w') as f:
            json.dump(configuration, f, indent=4)
        logger.info(f"Configuration successfully saved to {path}")
        return configuration
    except Exception as e:
        logger.error(f"Failed to save configuration to {path}: {e}")
        raise
