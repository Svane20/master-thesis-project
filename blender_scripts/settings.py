from pydantic import BaseModel
from enum import Enum
import json


class RenderType(str, Enum):
    Cycles = "CYCLES"
    Eevee = "BLENDER_EEVEE_NEXT"


class CameraSettings(BaseModel):
    camera_fov_mu_deg: float = 60.0
    camera_fov_std_deg: float = 10
    image_width: int = 2048
    image_height: int = 2048


class RenderSettings(BaseModel):
    gpu_indices: list = [0, 2]
    render: RenderType = RenderType.Cycles
    tmp_folder: str = "tmp"
    n_cycles: int = 128

    camera_cull_margin: float = 1.0
    distance_cull_margin: float = 200.0

    taa_render_samples: int = 64
    shadow_ray_count: int = 4
    shadow_step_count: int = 12


class Terrain(BaseModel):
    world_size: float = 100
    image_size: int = 2048
    prob_of_trees: float = 0.25


class Settings(BaseModel):
    render_settings: RenderSettings = RenderSettings()
    camera_settings: CameraSettings = CameraSettings()
    terrain: Terrain = Terrain()


def load_settings(path: str) -> dict:
    with open(path, 'r') as f:
        settings = json.load(f)

    return settings


def save_settings(settings: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(settings, f, indent=4)
