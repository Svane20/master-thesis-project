import bpy
from mathutils import Vector, Quaternion, Euler


class CameraConstants:
    CAMERA_NAME = "Camera"


def update_camera_position(location: Vector = None, focus_point: Vector = Vector((0.0, 0.0, 0.0))) -> bpy.types.Object:
    camera: bpy.types.Object = bpy.data.objects[CameraConstants.CAMERA_NAME]
    if camera is None:
        print("Camera not found")
        return

    if location is not None:
        camera.location = location

    current_direction = camera.location - focus_point
    rotation_quaternion: Quaternion = current_direction.to_track_quat("Z", "Y")
    camera.rotation_euler: Euler = rotation_quaternion.to_euler()

    return camera
