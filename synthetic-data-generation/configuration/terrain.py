from pydantic import BaseModel

from constants.defaults import ImageDefaults, WorldDefaults


class TerrainConfiguration(BaseModel):
    """
    Configuration for terrain generation in the scene.

    Attributes:
        world_size (float): The size of the terrain (world).
        image_size (int): The size of the image or terrain representation.
        prob_of_trees (float): The probability of placing trees in the terrain (value between 0 and 1).
    """
    world_size: float = WorldDefaults.SIZE
    image_size: int = ImageDefaults.SIZE
    prob_of_trees: float = 0.25
