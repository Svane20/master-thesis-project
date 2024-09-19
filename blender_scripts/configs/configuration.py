from pathlib import Path
from pydantic import BaseModel
from typing import List, Union

from enum import Enum
import json

from consts import Constants


class RenderType(str, Enum):
    Cycles = "CYCLES"
    Eevee = "BLENDER_EEVEE_NEXT"


class CameraConfiguration(BaseModel):
    camera_fov_mu_deg: float = 60.0
    camera_fov_std_deg: float = 10
    image_width: int = Constants.Default.IMAGE_WIDTH
    image_height: int = Constants.Default.IMAGE_HEIGHT


class RenderConfiguration(BaseModel):
    gpu_indices: List[int] = [0, 2]
    render: RenderType = RenderType.Cycles
    tmp_folder: str = Constants.Directory.TEMP_DIR.as_posix()
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


def load_configuration(path: Union[str, Path]) -> dict:
    """Load settings from a JSON file."""
    try:
        with Path(path).open('r') as f:
            settings = json.load(f)

        return settings
    except Exception as e:
        print(f"Failed to load settings: {e}")


def save_configuration(configuration: dict, path: Union[str, Path]) -> None:
    """Save settings to a JSON file."""
    try:
        with Path(path).open('w') as f:
            json.dump(configuration, f, indent=4)
    except Exception as e:
        print(f"Failed to save settings: {e}")
