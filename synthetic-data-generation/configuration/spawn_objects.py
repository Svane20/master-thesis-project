from pydantic import BaseModel
from typing import List


class SpawnObject(BaseModel):
    """
    Configuration for spawning an object

    Attributes:
        should_spawn (bool): Whether to spawn the object
        use_halton (bool): Whether to use Halton to set the position
        num_objects (int): Number of objects to spawn
        position (List[int]): The positions to spawn the objects in
        directory (str): The directory where to find objects to spawn
        include (List[str]): The keywords that must be present in the file path
        exclude (List[str]): The keywords to filter out from the file paths
    """
    should_spawn: bool
    use_halton: bool
    num_objects: int
    position: List[int] | None
    directory: str
    include: List[str] | None
    exclude: List[str] | None


class SpawnObjectsConfiguration(BaseModel):
    spawn_objects: List[SpawnObject]
