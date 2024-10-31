from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import random

from custom_logging.custom_logger import setup_logger
from bpy_utils.bpy_ops import append_object

logger = setup_logger(__name__)


def spawn_objects(
        num_objects: int,
        positions: NDArray[np.float32],
        path: Path,
        height_map: NDArray[np.float32],
        world_size: float,
        seed: int = None
) -> None:
    """
    Spawn objects on the terrain at specified positions.

    Args:
        num_objects (int): The number of objects to spawn.
        positions (NDArray[np.float32]): The (x, y) positions where objects will be placed.
        path (Path): The directory path containing .blend files for objects.
        height_map (NDArray[np.float32]): The terrain height map.
        world_size (float): The size of the world (terrain scale).
        seed (int, optional): Random seed for reproducibility. Default is None.

    Raises:
        FileNotFoundError: If no .blend files are found in the specified path.
    """
    logger.info(f"Spawning {num_objects} objects on the terrain.")

    # Ensure terrain dimensions match the expected format
    height, width = height_map.shape[:2]
    logger.debug(f"Terrain dimensions: width={width}, height={height}")

    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to {seed}")

    blend_objects_paths = list(path.rglob("*.blend"))
    if not blend_objects_paths:
        logger.error(f"No .blend files found in directory {path}")
        raise FileNotFoundError(f"No .blend files found in directory {path}")

    logger.info(f"Found {len(blend_objects_paths)} .blend files in {path}")

    for index in range(num_objects):
        random_object_path = random.choice(blend_objects_paths)
        logger.info(f"Selected object file: {random_object_path}")

        # Append the object to the scene
        collection_object = append_object(object_path=random_object_path)
        logger.info(f"Appended object from {random_object_path}")

        for obj in collection_object.objects:
            if obj.parent is not None:
                logger.debug(f"Skipping object '{obj.name}' as it has a parent.")
                continue

            # Get object position (x, y)
            x, y = positions[index, 0], positions[index, 1]
            logger.debug(f"Object {index + 1}: initial position ({x}, {y})")

            # Map normalized coordinates to terrain grid
            x_ = int((x / world_size + 0.5) * width)
            x_ = np.clip(x_, 0, width - 1)
            y_ = int((y / world_size + 0.5) * height)
            y_ = np.clip(y_, 0, height - 1)
            h = height_map[x_, y_]

            # Set object location, rotation, and pass index
            obj.location = (x, y, h)
            obj.rotation_euler = (0, 0, random.random() * np.pi * 2)
            obj.pass_index = 2

            logger.info(
                f"Placed object '{obj.name}' at position ({x:.2f}, {y:.2f}, {h:.2f}) with rotation {obj.rotation_euler}."
            )
