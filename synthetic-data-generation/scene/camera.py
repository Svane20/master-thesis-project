import bpy
from mathutils import Vector, Euler
import numpy as np
from numpy.typing import NDArray
import logging


def get_camera_iterations(
        start: float = 0.0,
        stop: float = 2 * np.pi,
        num_iterations: int = 8,
        endpoint: bool = False,
        seed: int = None
) -> NDArray[np.float64]:
    """
    Generate camera iterations for the camera locations.

    Args:
        start (float, optional): The start of the iteration. Defaults to 0.0.
        stop (float, optional): The end of the iteration. Defaults to 2 * np.pi.
        num_iterations (int, optional): The number of iterations. Defaults to 8.
        endpoint (bool, optional): If True, the stop value is included in the iterations. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        NDArray[np.float64]: The camera iterations.
    """
    if seed is not None:
        np.random.seed(seed)
        logging.info(f"Seed set to {seed}")

    return np.linspace(start=start, stop=stop, num=num_iterations, endpoint=endpoint)


def get_random_camera_location(iteration: float, height_map: NDArray[np.float32], world_size: int, seed: int = None) -> Vector:
    """
    Generate a random camera location.

    Args:
        iteration (float): The current iteration of the camera
        height_map (NDArray[np.float32]): The terrain height map.
        world_size (int): The size of the world.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Vector: The random camera location.
    """
    if seed is not None:
        np.random.seed(seed)
        logging.info(f"Seed set to {seed}")

    # Get the terrain shape
    width, height = height_map.shape[:2]

    # Get the maximum distance from the center of the world
    maximal_distance = world_size / 2

    # Calculate the x coordinate
    x = maximal_distance * np.cos(iteration)
    x = np.clip(a=x, a_min=-maximal_distance, a_max=maximal_distance)
    x_ = (x / world_size + 0.5) * width
    x_ = np.clip(a=x_, a_min=0, a_max=width - 1)

    # Calculate the y coordinate
    y = maximal_distance * np.sin(iteration)
    y = np.clip(a=y, a_min=-maximal_distance, a_max=maximal_distance)
    y_ = (y / world_size + 0.5) * height
    y_ = np.clip(a=y_, a_min=0, a_max=height - 1)

    # Calculate the height
    height = height_map[int(x_), int(y_)]

    return Vector((x, y, height + np.random.uniform(low=1.5, high=25)))


def update_camera_position(
        location: Vector = None,
        rotation: Vector = None,
        focus_point: Vector = Vector((0.0, 0.0, 0.0))
) -> bpy.types.Object:
    """
    Update the camera's location and rotation.

    Args:
        location (Vector, optional): The camera location. If not provided, the current location is used.
        rotation (Vector, optional): The camera rotation in Euler angles. If not provided, the camera will focus on the focus_point.
        focus_point (Vector, optional): The point the camera is focusing on. Used if rotation is not provided.

    Returns:
        bpy.types.Object: The updated camera object.

    Raises:
        ValueError: If the camera is not found.
    """
    try:
        camera: bpy.types.Object = bpy.data.objects["Camera"]
    except KeyError:
        logging.error("Camera object not found in the scene.")
        raise ValueError("Camera not found")

    if location is not None:
        logging.info(f"Updating camera location to {location}.")
        camera.location = location
    else:
        logging.info("No location provided; using the current camera location.")

    if rotation is not None:
        logging.info(f"Updating camera rotation to {rotation}.")
        camera.rotation_euler = rotation
    else:
        logging.info(f"No rotation provided; calculating rotation to focus on {focus_point}.")
        camera.rotation_euler = _calculate_rotation_to_focus(camera.location, focus_point)

    logging.info(
        "Camera updated successfully",
        extra={
            "location": f"({camera.location.x:.2f}, {camera.location.y:.2f}, {camera.location.z:.2f})",
            "rotation": f"(x={camera.rotation_euler.x:.2f}, y={camera.rotation_euler.y:.2f}, z={camera.rotation_euler.z:.2f})",
            "focus_point": f"({focus_point.x:.2f}, {focus_point.y:.2f}, {focus_point.z:.2f})"
        }
    )

    return camera


def _calculate_rotation_to_focus(location: Vector, focus_point: Vector) -> Euler:
    """
    Calculate the rotation required for the camera to focus on a given point.

    Args:
        location (Vector): The current location of the camera.
        focus_point (Vector): The point the camera should focus on.

    Returns:
        Euler: The calculated rotation in Euler angles.
    """
    current_direction: Vector = location - focus_point
    return current_direction.to_track_quat("Z", "Y").to_euler()
