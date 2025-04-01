from pathlib import Path
from pydantic import BaseModel
from typing import Union, Dict
import json
from typing import List
import platform

from configuration.addons import AddonConfiguration
from configuration.camera import CameraConfiguration
from configuration.constants import Constants
from configuration.run import RunConfiguration
from configuration.sky import SkyConfiguration
from configuration.render import RenderConfiguration
from configuration.spawn_objects import SpawnObjectsConfiguration
from configuration.terrain import TerrainConfiguration


class Configuration(BaseModel):
    """
    The Base class for all configurations.

    Attributes:
        addons (List[AddonConfiguration]): A list of addon configurations.
        constants (List[Constant]): A list of constant configurations.
        render_configuration (RenderConfiguration): Render configuration.
        camera_configuration (CameraConfiguration): Camera configuration.
        terrain_configuration (TerrainConfiguration): Terrain configuration.
        sky_configuration (SkyConfiguration): HDRI configuration.
    """
    addons: List[AddonConfiguration]
    constants: Constants
    run_configuration: RunConfiguration
    render_configuration: RenderConfiguration
    camera_configuration: CameraConfiguration
    terrain_configuration: TerrainConfiguration
    spawn_objects_configuration: SpawnObjectsConfiguration
    sky_configuration: SkyConfiguration


def get_configuration(is_colab: bool = False) -> Configuration:
    """
    Loads the configuration for the Blender pipeline.

    Args:
        is_colab (bool): Whether the pipeline is running in Google Colab. Defaults to False.

    Returns:
        Configuration: The configuration for the Blender pipeline

    """
    # Detect OS and set the configuration path accordingly
    base_directory = Path(__file__).resolve().parent.parent
    configs_directory = base_directory / "configs"

    if platform.system() == "Windows":
        configuration_path: Path = configs_directory / "configuration_windows.json"
    elif is_colab:
        configuration_path: Path = configs_directory / "configuration_colab.json"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = configs_directory / "configuration_linux.json"

    config = _load_configuration(configuration_path)

    return Configuration(**config)


def _load_configuration(path: Path) -> Union[Dict[str, Union[str, int, float, bool, dict]], None]:
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
        print(f"Configuration successfully loaded from {path}")
        return configuration
    except FileNotFoundError as e:
        print(f"Configuration file not found: {path}. Error: {e}")
        raise e
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in configuration file: {path}. Error: {e}")
        raise e
    except Exception as e:
        print(f"Failed to load configuration from {path}: {e}")
        raise e
