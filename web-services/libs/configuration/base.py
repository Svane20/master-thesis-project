from typing import Tuple, List

from pydantic import BaseModel, Field
import json


class ProjectInfoConfiguration(BaseModel):
    project_name: str = Field(...)
    model_type: str = Field(...)


class ModelTransformsConfiguration(BaseModel):
    image_size: Tuple[int, int] = Field(...)
    mean: List[float] = Field(...)
    std: List[float] = Field(...)


class ModelConfiguration(BaseModel):
    model_path: str = Field(...)
    transforms: ModelTransformsConfiguration


class Configuration(BaseModel):
    project_info: ProjectInfoConfiguration
    model: ModelConfiguration


def get_configuration(configuration_path: str = "./configs/config.json") -> Configuration:
    """
    Loads the configuration from a JSON file.

    Args:
        configuration_path (str): The path to the configuration file.

    Returns:
        Configuration: The loaded configuration object.
    """
    try:
        with open(configuration_path, 'r') as file:
            config = json.load(file)
        return Configuration(**config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {configuration_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from configuration file: {configuration_path}") from e
    except TypeError as e:
        raise ValueError(f"Error loading configuration: {e}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e
