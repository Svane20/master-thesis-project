from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import random
import logging
from typing import List

from bpy_utils.bpy_ops import append_object


def spawn_objects(
        num_objects: int,
        positions: NDArray[np.float32],
        filepath: str,
        height_map: NDArray[np.float32],
        world_size: float,
        keywords: List[str] | None = None,
        seed: int = None
) -> None:
    """
    Spawn objects on the terrain at specified positions.

    Args:
        num_objects (int): The number of objects to spawn.
        positions (NDArray[np.float32]): The (x, y) positions where objects will be placed.
        filepath (str): The directory path containing .blend files for objects.
        height_map (NDArray[np.float32]): The terrain height map.
        world_size (float): The size of the world (terrain scale).
        keywords (List[str], optional): A list of keywords to filter object files. Default is None.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Raises:
        FileNotFoundError: If no .blend files are found in the specified path.
    """
    path = Path(filepath)

    logging.info(f"Spawning {num_objects} {path.name} on the terrain...")

    # Ensure terrain dimensions match the expected format
    height, width = height_map.shape[:2]
    logging.debug(f"Terrain dimensions: width={width}, height={height}")

    if seed is not None:
        random.seed(seed)
        logging.debug(f"Random seed set to {seed}")

    blend_objects_paths = list(path.rglob("*.blend"))
    if not blend_objects_paths:
        logging.error(f"No .blend files found in directory {path}")
        raise FileNotFoundError(f"No .blend files found in directory {path}")

    if keywords:
        blend_objects_paths = [path for path in blend_objects_paths if
                               any(keyword in path.name for keyword in keywords)]

    logging.debug(f"Found {len(blend_objects_paths)} .blend files in {path}")

    for index in range(num_objects):
        random_object_path = random.choice(blend_objects_paths)
        logging.debug(f"Selected object file: {random_object_path}")

        # Append the object to the scene
        collection_object = append_object(object_path=random_object_path)
        logging.debug(f"Appended object from {random_object_path}")

        for obj in collection_object.objects:
            if obj.parent is not None:
                logging.debug(f"Skipping object '{obj.name}' as it has a parent.")
                continue

            # Get object position (x, y)
            x, y = positions[index, 0], positions[index, 1]
            logging.debug(f"Object {index + 1}: initial position ({x}, {y})")

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

            logging.debug(
                f"Placed object '{obj.name}' at position ({x:.2f}, {y:.2f}, {h:.2f}) with rotation {obj.rotation_euler}."
            )

    logging.info(f"Finished spawning {num_objects} {path.name} on the terrain")