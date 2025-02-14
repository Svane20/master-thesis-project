from pathlib import Path
from pydantic import BaseModel
from typing import Union, Dict
import json
from typing import List
import logging

from configuration.addons import AddonConfiguration
from configuration.camera import CameraConfiguration
from configuration.constants import Constants
from configuration.directories import Directories
from configuration.hdri import HDRIConfiguration
from configuration.render import RenderConfiguration
from configuration.terrain import TerrainConfiguration


class Configuration(BaseModel):
    """
    The Base class for all configurations.

    Attributes:
        addons (List[AddonConfiguration]): A list of addon configurations.
        constants (List[Constant]): A list of constant configurations.
        directories (List[Directories]): A list of directories configurations.
        render_configuration (RenderConfiguration): Render configuration.
        camera_configuration (CameraConfiguration): Camera configuration.
        terrain_configuration (TerrainConfiguration): Terrain configuration.
        hdri_configuration (HDRIConfiguration): HDRI configuration.
    """
    addons: List[AddonConfiguration]
    constants: Constants
    directories: Directories
    render_configuration: RenderConfiguration
    camera_configuration: CameraConfiguration
    terrain_configuration: TerrainConfiguration
    hdri_configuration: HDRIConfiguration


def load_configuration(path: Path) -> Union[Dict[str, Union[str, int, float, bool, dict]], None]:
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
        logging.info(f"Configuration successfully loaded from {path}")
        return configuration
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {path}. Error: {e}")
        raise e
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in configuration file: {path}. Error: {e}")
        raise e
    except Exception as e:
        logging.error(f"Failed to load configuration from {path}: {e}")
        raise e
