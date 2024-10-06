from pathlib import Path
from pydantic import BaseModel
from typing import Union

from enum import Enum
import json

from configuration.consts import Constants
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


class RenderType(str, Enum):
    Cycles = "CYCLES"
    Eevee = "BLENDER_EEVEE_NEXT"
    Workbench = "BLENDER_WORKBENCH"


class CameraConfiguration(BaseModel):
    camera_fov_mu_deg: float = 60.0
    camera_fov_std_deg: float = 10
    image_width: int = Constants.Default.IMAGE_WIDTH
    image_height: int = Constants.Default.IMAGE_HEIGHT


class RenderConfiguration(BaseModel):
    render: RenderType = RenderType.Eevee
    temp_folder: str = Constants.Directory.TEMP_DIR.as_posix()
    n_cycles: int = 128

    camera_cull_margin: float = 1.0
    distance_cull_margin: float = 200.0

    taa_render_samples: int = 64
    shadow_ray_count: int = 4
    shadow_step_count: int = 12


class TerrainConfiguration(BaseModel):
    world_size: float = Constants.Default.WORLD_SIZE
    image_size: int = Constants.Default.IMAGE_SIZE
    prob_of_trees: float = 0.25


class Configuration(BaseModel):
    render_configuration: RenderConfiguration = RenderConfiguration()
    camera_configuration: CameraConfiguration = CameraConfiguration()
    terrain_configuration: TerrainConfiguration = TerrainConfiguration()


def load_configuration(path: Union[str, Path]) -> Union[dict, None]:
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


def save_configuration(configuration: dict, path: Union[str, Path]) -> dict:
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
