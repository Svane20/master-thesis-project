from pydantic import BaseModel
import logging
import json
from pathlib import Path
from typing import Dict, Union
import platform


class Configuration(BaseModel):
    """
    Configuration for the dataset conversion pipeline.

    Attributes:
        source_directory (str): The directory containing the source dataset.
        destination_directory (str): The directory where the converted dataset will be saved.
    """
    source_directory: str
    destination_directory: str


def get_configurations() -> Configuration:
    """
    Loads the configuration for the dataset conversion pipeline.

    Returns:
        Configuration: The configuration for the dataset conversion pipeline
    """
    base_directory = Path(__file__).resolve().parent.parent

    # Load the configuration file based on the OS
    if platform.system() == "Windows":
        configuration_path: Path = base_directory / "configuration_windows.json"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = base_directory / "configuration_linux.json"

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
