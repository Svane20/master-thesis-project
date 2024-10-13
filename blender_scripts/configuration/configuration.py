from pathlib import Path
from pydantic import BaseModel
from typing import Union, Dict
import json

from configuration.camera import CameraConfiguration
from configuration.hdri import HDRIConfiguration
from configuration.render import RenderConfiguration
from configuration.terrain import TerrainConfiguration

from constants.directories import CONFIG_PATH
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


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
        FileNotFoundError: If the configuration file is not found.
        JSONDecodeError: If the file contains invalid JSON.
        Exception: For any other unexpected errors.
    """
    try:
        with path.open('r') as f:
            configuration = json.load(f)
        logger.info(f"Configuration successfully loaded from {path}")
        return configuration
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {path}. Error: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in configuration file: {path}. Error: {e}")
        return None
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
