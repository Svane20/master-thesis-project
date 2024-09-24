import bpy
from mathutils import Vector, Euler


class CameraConstants:
    CAMERA_NAME = "Camera"


def update_camera_position(
        location: Vector = None,
        rotation: Vector = None,
        focus_point: Vector = Vector((0.0, 0.0, 0.0))
) -> bpy.types.Object:
    camera: bpy.types.Object = bpy.data.objects[CameraConstants.CAMERA_NAME]
    if camera is None:
        print("Camera not found")
        return

    if location is not None:
        camera.location = location

    if rotation is not None:
        camera.rotation_euler = rotation
    else:
        current_direction: Vector = camera.location - focus_point
        camera.rotation_euler: Euler = current_direction.to_track_quat("Z", "Y").to_euler()

    return camera
