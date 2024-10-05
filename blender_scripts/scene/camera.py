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
        location: The camera location.
        rotation: The camera rotation.
        focus_point: The focus point.

    Returns:
        The camera object.

    Raises:
        ValueError: If the camera is not found.
    """
    camera: bpy.types.Object = bpy.data.objects["Camera"]
    if camera is None:
        logger.error("Camera not found")
        raise ValueError("Camera not found")

    if location is not None:
        camera.location = location

    if rotation is not None:
        camera.rotation_euler = rotation
    else:
        current_direction: Vector = camera.location - focus_point
        camera.rotation_euler: Euler = current_direction.to_track_quat("Z", "Y").to_euler()

    logger.info(
        "Camera updated",
        extra={
            "location": f"({camera.location.x:.2f}, {camera.location.y:.2f}, {camera.location.z:.2f})",
            "rotation": f"(x={camera.rotation_euler.x:.2f}, y={camera.rotation_euler.y:.2f}, z={camera.rotation_euler.z:.2f})",
            "focus_point": f"({focus_point.x:.2f}, {focus_point.y:.2f}, {focus_point.z:.2f})"
        }
    )

    return camera
