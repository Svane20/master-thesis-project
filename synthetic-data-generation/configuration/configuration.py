from pathlib import Path
from pydantic import BaseModel
from typing import Union, Dict
import json
from typing import List
import platform

from configuration.addons import AddonConfiguration
from configuration.camera import CameraConfiguration
from configuration.constants import Constants
from configuration.directories import Directories
from configuration.hdri import HDRIConfiguration
from configuration.render import RenderConfiguration
from configuration.terrain import TerrainConfiguration
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)

# Detect OS and set the configuration path accordingly
BASE_DIRECTORY: Path = Path(__file__).resolve().parent.parent
if platform.system() == "Windows":
    CONFIG_PATH: Path = BASE_DIRECTORY / "configuration_windows.json"
else:  # Assume Linux for any non-Windows OS
    CONFIG_PATH: Path = BASE_DIRECTORY / "configuration_linux.json"


class Configuration(BaseModel):
    addons: List[AddonConfiguration]
    constants: Constants
    directories: Directories
    render_configuration: RenderConfiguration
    camera_configuration: CameraConfiguration
    terrain_configuration: TerrainConfiguration
    hdri_configuration: HDRIConfiguration


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
        raise e
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in configuration file: {path}. Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}")
        raise e
