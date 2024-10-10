from pathlib import Path

import numpy as np
import random

from custom_logging.custom_logger import setup_logger
from bpy_utils.bpy_ops import append_object

logger = setup_logger(__name__)


def spawn_objects(
        num_objects: int,
        positions: np.ndarray,
        path: Path,
        terrain: np.ndarray,
        world_size: float,
        seed: int = None
) -> None:
    """
    Spawn objects on the terrain.

    Args:
        num_objects: The number of objects to spawn.
        positions: The positions of the objects.
        path: The path to the objects.
        terrain: The terrain.
        world_size: The world size.
        seed: Random seed for reproducibility. Default is None.
    """
    logger.info(f"Spawning {num_objects} objects")

    height, width = terrain.shape[:2]

    if seed is not None:
        random.seed(seed)

    for index in range(num_objects):
        blend_objects_paths = list(path.rglob("*.blend"))
        random_object_path = random.choice(blend_objects_paths)

        # Append the object to the scene
        collection_object = append_object(object_path=random_object_path)

        for obj in collection_object.objects:
            if obj.parent is not None:
                continue

            x, y = positions[index, 0], positions[index, 1]

            # Map normalized coordinates to terrain grid
            x_ = int((x / world_size + 0.5) * width)
            x_ = np.clip(x_, 0, width - 1)

            y_ = int((y / world_size + 0.5) * height)
            y_ = np.clip(y_, 0, height - 1)

            h = terrain[x_, y_]

            # Set object location, rotation and pass index
            obj.location = (x, y, h)
            obj.rotation_euler = (0, 0, random.random() * np.pi * 2)
            obj.pass_index = 2

            logger.info(f"Set object '{obj.name}' at position ({x}, {y}, {h})")
