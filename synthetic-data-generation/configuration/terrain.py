from pydantic import BaseModel
from typing import List


class TreesConfiguration(BaseModel):
    directory: str
    keywords: List[str] | None


class GrassConfiguration(BaseModel):
    directory: str
    keywords: List[str] | None


class NotGrassConfiguration(BaseModel):
    directory: str
    keywords: List[str] | None


class TerrainConfiguration(BaseModel):
    """
    Configuration for terrain generation in the scene.

    Attributes:
        world_size (float): The size of the terrain (world).
        image_size (int): The size of the image or terrain representation.
        prob_of_trees (float): The probability of placing trees in the terrain (value between 0 and 1).
    """
    world_size: float
    image_size: int
    noise_basis: str
    prob_of_trees: float
    trees_configuration: TreesConfiguration
    grass_configuration: GrassConfiguration
    not_grass_configuration: NotGrassConfiguration
