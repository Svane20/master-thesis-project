import bpy
from mathutils import Vector, Euler

from enum import Enum


class LightType(Enum):
    AREA = "AREA"
    POINT = "POINT"
    SPOT = "SPOT"
    SUN = "SUN"


def _delete_all_lights():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()


def create_light(
        light_name: str,
        light_type: LightType,
        energy: float,
        location: Vector = Vector((0.0, 0.0, 0.0)),
        rotation: Euler = Euler((0.0, 0.0, 0.0)),
        use_shadow: bool = True,
        specular_factor: float = 1.0,
) -> bpy.types.Object:
    _delete_all_lights()

    # Create a new light and the corresponding object
    new_light: bpy.types.Light = bpy.data.lights.new(name=light_name, type=light_type.value)
    new_light_object: bpy.types.Object = bpy.data.objects.new(name=light_name, object_data=new_light)

    # Add the light object to the scene
    bpy.context.collection.objects.link(new_light_object)

    # Set bpy.types.Object properties
    new_light_object.location = location
    new_light_object.rotation_euler = rotation

    # Set bpy.types.Light properties
    new_light.use_shadow = use_shadow
    new_light.specular_factor = specular_factor
    new_light.energy = energy

    return new_light_object
