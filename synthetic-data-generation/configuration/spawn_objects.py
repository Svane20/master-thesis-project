from pydantic import BaseModel
from typing import List


class SpawnObject(BaseModel):
    """
    Configuration for spawning a object

    Attributes:
        should_spawn (bool): Whether to spawn the object
        use_halton (bool): Whether to use Halton to set the position
        num_objects (int): Number of objects to spawn
        position (List[int]): The positions to spawn the objects in
        directory (str): The directory where to find objects to spawn
    """
    should_spawn: bool
    use_halton: bool
    num_objects: int
    position: List[int] | None
    directory: str


class SpawnObjectsConfiguration(BaseModel):
    spawn_objects: List[SpawnObject]
