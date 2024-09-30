import bpy
from mathutils import Vector, Euler

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def update_camera_position(
        location: Vector = None,
        rotation: Vector = None,
        focus_point: Vector = Vector((0.0, 0.0, 0.0))
) -> bpy.types.Object:
    """Update the camera's location and rotation."""
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

    logger.info(f"Camera has been updated to location: {camera.location}, rotation: {camera.rotation_euler}")

    return camera
