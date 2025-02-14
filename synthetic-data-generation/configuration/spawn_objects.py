from pydantic import BaseModel
from typing import List


class HousesObjectConfiguration(BaseModel):
    """
    Configuration for spawning a house

    Attributes:
        should_spawn (bool): Whether to spawn the object
        num_objects (int): Number of objects to spawn
        position (List[int]): The positions to spawn the objects in
        directory (str): The directory where to find objects to spawn
    """
    should_spawn: bool
    num_objects: int
    position: List[int]
    directory: str


class SpawnObjectsConfiguration(BaseModel):
    houses_configuration: HousesObjectConfiguration
