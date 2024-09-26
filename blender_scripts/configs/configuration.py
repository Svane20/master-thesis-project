from pathlib import Path
from pydantic import BaseModel
from typing import Union

from enum import Enum
import json

from consts import Constants


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
    """Load settings from a JSON file."""
    try:
        with Path(path).open('r') as f:
            configuration = json.load(f)

        return configuration
    except Exception as e:
        print(f"Configuration have not been set: {e}")
        return None


def save_configuration(configuration: dict, path: Union[str, Path]) -> dict:
    """Save settings to a JSON file."""
    try:
        with Path(path).open('w') as f:
            json.dump(configuration, f, indent=4)

        return configuration
    except Exception as e:
        print(f"Failed to save settings: {e}")
