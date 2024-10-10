from pathlib import Path
from pydantic import BaseModel
from typing import Union

from enum import Enum
import json

from constants.defaults import IMAGE_WIDTH, IMAGE_HEIGHT, WORLD_SIZE, IMAGE_SIZE
from constants.directories import TEMP_DIRECTORY, CONFIG_PATH
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


class EngineType(str, Enum):
    Cycles = "CYCLES"
    Eevee = "BLENDER_EEVEE_NEXT"
    Workbench = "BLENDER_WORKBENCH"


class CameraConfiguration(BaseModel):
    image_width: int = IMAGE_WIDTH
    image_height: int = IMAGE_HEIGHT

    camera_fov_mu_deg: float = 60.0
    camera_fov_std_deg: float = 10


class PreferencesConfiguration(BaseModel):
    compute_device_type: str = "CUDA"


class CyclesConfiguration(BaseModel):
    camera_cull_margin: float = 1.0
    distance_cull_margin: float = 200.0
    use_camera_cull: bool = True
    use_distance_cull: bool = True

    feature_set: str = "SUPPORTED"
    device: str = "GPU"
    tile_size: int = 4096
    # samples: int = 128 # Use in production
    samples: int = 1  # Use in development
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
    use_persistent_data: bool = True  # This helps reuse data between renders, reducing computation time
    threads_mode: str = "FIXED"
    threads: int = 54
    compression: int = 0

    cycles_configuration: CyclesConfiguration = CyclesConfiguration()
    preferences_configuration: PreferencesConfiguration = PreferencesConfiguration()


class TerrainConfiguration(BaseModel):
    world_size: float = WORLD_SIZE
    image_size: int = IMAGE_SIZE
    prob_of_trees: float = 0.25


class Configuration(BaseModel):
    render_configuration: RenderConfiguration = RenderConfiguration()
    camera_configuration: CameraConfiguration = CameraConfiguration()
    terrain_configuration: TerrainConfiguration = TerrainConfiguration()


def load_configuration(path: Path = CONFIG_PATH) -> Union[dict, None]:
    """
    Load settings from a JSON file.

    Args:
        path: The path to the JSON file.

    Returns:
        The settings as a dictionary

    Raises:
        Exception: If the settings could not be loaded.
    """
    try:
        with Path(path).open('r') as f:
            configuration = json.load(f)

        return configuration
    except Exception as e:
        logger.info(f"Configuration have not been set: {e}")
        return None


def save_configuration(configuration: dict, path: Path = CONFIG_PATH) -> dict:
    """
    Save settings to a JSON file.

    Args:
        configuration: The settings as a dictionary.
        path: The path to the JSON file.

    Returns:
        The settings as a dictionary.

    Raises:
        Exception: If the settings could not be saved.
    """
    try:
        with Path(path).open('w') as f:
            json.dump(configuration, f, indent=4)

        return configuration
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        raise
