import bpy
from mathutils import Vector, Euler
from enum import Enum
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


class LightType(str, Enum):
    """
    Enumeration for light types in Blender.

    Attributes:
        AREA: Area light type.
        POINT: Point light type.
        SPOT: Spotlight type.
        SUN: Sunlight type.
    """
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
        light_name (str): The name of the light.
        light_type (LightType): The type of light (AREA, POINT, SPOT, SUN).
        energy (float): The intensity of the light.
        location (Vector, optional): The location of the light in the scene. Defaults to the origin.
        rotation (Euler, optional): The rotation of the light. Defaults to no rotation.
        use_shadow (bool, optional): Enable or disable shadows for the light. Defaults to True.
        specular_factor (float, optional): The specular intensity for the light. Defaults to 1.0.
        scene (bpy.types.Scene, optional): The scene where the light will be created. Defaults to the current scene.
        delete_existing_lights (bool, optional): If True, delete all existing lights in the scene. Defaults to False.

    Returns:
        bpy.types.Object: The newly created light object.

    Raises:
        Exception: If the light creation fails.
    """

    if location is None:
        location = Vector((0.0, 0.0, 0.0))
    if rotation is None:
        rotation = Euler((0.0, 0.0, 0.0))
    if scene is None:
        scene = bpy.context.scene

    if delete_existing_lights:
        logger.info("Deleting all existing lights.")
        _delete_all_lights()

    try:
        logger.info(f"Creating light '{light_name}' of type '{light_type.value}' with energy {energy}.")
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

        logger.info(f"Light '{light_name}' created successfully.", extra={
            "location": f"({new_light_object.location.x:.2f}, {new_light_object.location.y:.2f}, {new_light_object.location.z:.2f})",
            "rotation": f"(x={new_light_object.rotation_euler.x:.2f}, y={new_light_object.rotation_euler.y:.2f}, z={new_light_object.rotation_euler.z:.2f})",
            "shadow": use_shadow,
            "specular_factor": specular_factor,
            "energy": energy
        })

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
    try:
        lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
        logger.info(f"Deleting {len(lights)} lights from the scene.")

        for light in lights:
            bpy.data.objects.remove(light, do_unlink=True)

        logger.info("All lights have been deleted successfully.")
    except Exception as e:
        logger.error(f"Failed to delete all lights: {e}")
        raise
