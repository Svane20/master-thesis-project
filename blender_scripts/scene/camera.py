import bpy
from mathutils import Vector, Euler
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


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
        logger.error("Camera object not found in the scene.")
        raise ValueError("Camera not found")

    if location is not None:
        logger.info(f"Updating camera location to {location}.")
        camera.location = location
    else:
        logger.info("No location provided; using the current camera location.")

    if rotation is not None:
        logger.info(f"Updating camera rotation to {rotation}.")
        camera.rotation_euler = rotation
    else:
        logger.info(f"No rotation provided; calculating rotation to focus on {focus_point}.")
        camera.rotation_euler = _calculate_rotation_to_focus(camera.location, focus_point)

    logger.info(
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
