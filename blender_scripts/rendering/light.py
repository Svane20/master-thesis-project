import bpy
from mathutils import Vector, Euler

from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LightType(Enum):
    AREA = "AREA"
    POINT = "POINT"
    SPOT = "SPOT"
    SUN = "SUN"


def create_light(
        light_name: str,
        light_type: LightType,
        energy: float,
        location: Vector = None,
        rotation: Euler = None,
        use_shadow: bool = True,
        specular_factor: float = 1.0,
        scene: bpy.types.Scene = None,
        delete_existing_lights: bool = False
) -> bpy.types.Object:
    """
    Creates a new light in the scene.

    Args:
        light_name: The name of the light.
        light_type: The type of light.
        energy: The light energy.
        location: The light location.
        rotation: The light rotation.
        use_shadow: Whether to use shadows.
        specular_factor: The specular factor.
        scene: The scene.
        delete_existing_lights: Whether to delete existing lights.

    Returns:
        The light object.

    Raises:
        Exception: If the light fails to create.
    """

    if location is None:
        location = Vector((0.0, 0.0, 0.0))

    if rotation is None:
        rotation = Euler((0.0, 0.0, 0.0))

    if scene is None:
        scene = bpy.context.scene

    if delete_existing_lights:
        _delete_all_lights()

    try:
        data_lights = bpy.data.lights
        data_objects = bpy.data.objects

        # Create a new light data block
        new_light = data_lights.new(name=light_name, type=light_type.value)

        # Create a new object with the light data block
        new_light_object = data_objects.new(name=light_name, object_data=new_light)

        # Link the object to the scene's collection
        scene.collection.objects.link(new_light_object)

        # Set object properties
        new_light_object.location = location
        new_light_object.rotation_euler = rotation

        # Set light properties
        new_light.use_shadow = use_shadow
        new_light.specular_factor = specular_factor
        new_light.energy = energy

        logger.info(f"Light '{light_name}' created successfully.")
        return new_light_object

    except Exception as e:
        logger.error(f"Failed to create light '{light_name}': {e}")
        raise


def _delete_all_lights() -> None:
    """
    Deletes all lights in the scene.

    Raises:
        Exception: If the lights fail to delete.
    """
    lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']

    for light in lights:
        bpy.data.objects.remove(light, do_unlink=True)

    logger.info("All lights have been deleted.")
