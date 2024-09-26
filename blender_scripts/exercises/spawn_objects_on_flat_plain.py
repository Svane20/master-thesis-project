import bpy
from mathutils import Vector

import numpy as np
import random
from math import radians

from bpy_ops import render_image, save_as_blend_file
from consts import Constants
from main import setup
from rendering.camera import update_camera_position
from rendering.light import create_light, LightType
from utils import remove_temporary_files

NUM_CYLINDER_OBJECTS = 2
NUM_CUBE_OBJECTS = 2
NUM_CONE_OBJECTS = 2
PLANE_SIZE = 10

RENDERED_IMAGE_NAME = "spawn_objects_on_flat_plain"

CAMERA_ANGLES = [
    {"location": Vector((14.0, 0.0, 11.0)), "rotation": Vector((radians(50), radians(0), radians(90)))},
    {"location": Vector((14.0, 10.0, 11.0)), "rotation": Vector((radians(55), radians(0), radians(126)))},
    {"location": Vector((0.0, 0.0, 25)), "rotation": Vector((radians(0), radians(0), radians(0)))},
    {"location": Vector((-19.0, 0.0, 2.0)), "rotation": Vector((radians(90), radians(0), radians(-90)))}
]


def _remove_default_objects() -> None:
    """Remove default objects from the scene"""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def _set_scene() -> None:
    # Add a light to the scene
    create_light(
        light_name="Sun",
        light_type=LightType.SUN,
        energy=5.0,
    )


def spawn_random_cones(num_objects: int, add_rotation: bool = False, plane_size: float = PLANE_SIZE) -> None:
    _spawn_random_primitives(
        num_objects=num_objects,
        primitive_function=bpy.ops.mesh.primitive_cone_add,
        add_rotation=add_rotation,
        plane_size=plane_size,
        scale_range=(0.5, 1.5),
        primitive_name="Cone",
        additional_args={"vertices": 32, "radius1": 1.0, "depth": 2.0}
    )


def spawn_random_cylinders(num_objects: int, add_rotation: bool = False, plane_size: float = PLANE_SIZE) -> None:
    _spawn_random_primitives(
        num_objects=num_objects,
        primitive_function=bpy.ops.mesh.primitive_cylinder_add,
        add_rotation=add_rotation,
        plane_size=plane_size,
        scale_range=(0.5, 1.5),
        primitive_name="Cylinder",
        additional_args={"radius": 1.0, "depth": 2.0}
    )


def spawn_random_cubes(num_objects: int, add_rotation: bool = False, plane_size: float = PLANE_SIZE) -> None:
    _spawn_random_primitives(
        num_objects=num_objects,
        primitive_function=bpy.ops.mesh.primitive_cube_add,
        add_rotation=add_rotation,
        plane_size=plane_size,
        scale_range=(0.5, 1.0),
        primitive_name="Cube",
        additional_args={"size": 2.0}
    )


def _spawn_random_primitives(
        num_objects: int,
        primitive_function,
        add_rotation: bool,
        plane_size: float,
        scale_range: tuple,
        primitive_name: str,
        additional_args: dict
) -> None:
    # Generate random positions within the plane boundaries
    half_size = plane_size / 2

    for index in range(num_objects):
        # Randomize scale
        scale = random.uniform(*scale_range)

        # Calculate the maximum offset to ensure the object is fully on the plane
        max_offset = half_size - (1.5 * scale)

        # Generate random positions within adjusted boundaries
        x_position = random.uniform(-max_offset, max_offset)
        y_position = random.uniform(-max_offset, max_offset)

        # Random rotation around x, y, and z axes
        if add_rotation:
            rotation_x = random.uniform(0, 2 * np.pi)  # 0 to 360 degrees in radians
            rotation_y = random.uniform(0, 2 * np.pi)
            rotation_z = random.uniform(0, 2 * np.pi)
        else:
            rotation_x = rotation_y = rotation_z = 0

        rotation = (rotation_x, rotation_y, rotation_z)

        # Prepare parameters for primitive function
        primitive_params = {
            "enter_editmode": False,
            "align": 'WORLD',
            "location": (x_position, y_position, 0),
            "rotation": rotation,
        }

        # Update additional arguments for the specific primitive
        for key, value in additional_args.items():
            # Multiply scale factor for size-related parameters
            if key in ["radius", "radius1", "size", "depth"]:
                primitive_params[key] = value * scale
            else:
                primitive_params[key] = value

        # Spawn the primitive object
        primitive_function(**primitive_params)

        # Get the reference to the newly created object
        obj: bpy.types.Object = bpy.context.object
        obj.name = f"{primitive_name}_{index}"

        # Adjust the object's position so it is resting on the ground
        place_object_on_ground(obj)


def place_object_on_ground(obj: bpy.types.Object):
    """Adjust the object location to ensure it's resting on the ground."""
    # Update the object's bounding box
    bpy.context.view_layer.update()

    # Calculate the lowest Z point of the object's vertices after transformation
    global_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_z = min(vertex.z for vertex in global_vertices)

    # Adjust the object's Z location so the lowest point is at Z = 0
    obj.location.z -= min_z


def render_from_angles(image_name: str, angles):
    """Render images from multiple camera angles."""
    output_dir = Constants.Directory.OUTPUT_DIR
    blender_dir = Constants.Directory.BLENDER_FILES_DIR

    for i, angle in enumerate(angles):
        # Update camera position and rotation
        bpy.context.scene.camera = update_camera_position(
            location=angle['location'],
            rotation=angle['rotation']
        )

        unique_image_name = f"{image_name}_{i + 1}"

        # Set the render filepath to the desired output path with the unique name
        bpy.context.scene.render.filepath = str(output_dir / f"{unique_image_name}.{Constants.Default.IMAGE_FORMAT}")

        # Render and save the image
        render_image()

        # Save the current scene as a .blend file
        save_as_blend_file(unique_image_name, blender_dir)


def main() -> None:
    setup(RENDERED_IMAGE_NAME)

    _set_scene()

    # Add a plane to the scene
    bpy.ops.mesh.primitive_plane_add(
        size=PLANE_SIZE,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0)
    )

    spawn_random_cones(num_objects=NUM_CONE_OBJECTS, add_rotation=True)
    spawn_random_cylinders(num_objects=NUM_CYLINDER_OBJECTS, add_rotation=False)
    spawn_random_cubes(num_objects=NUM_CUBE_OBJECTS, add_rotation=True)

    render_from_angles(RENDERED_IMAGE_NAME, CAMERA_ANGLES)

    # Remove temporary files
    remove_temporary_files(directory=Constants.Directory.OUTPUT_DIR, image_name=f"{RENDERED_IMAGE_NAME}0001")
    remove_temporary_files(directory=Constants.Directory.TEMP_DIR)


if __name__ == "__main__":
    main()
