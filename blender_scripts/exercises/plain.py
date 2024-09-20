import bpy
from mathutils import Vector

from camera import update_camera_position
from main import setup
from utils import save_image_file

def remove_default_objects() -> None:
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def main() -> None:
    setup(output_name="plain")

    remove_default_objects()

    # 1. Add a plane to the scene
    bpy.ops.mesh.primitive_plane_add(size=3, enter_editmode=False, align='WORLD', location=(0, 0, 0))

    # Step 2: Update camera position
    bpy.context.scene.camera = update_camera_position(location=Vector((0, -5, 5)))

    # Step 3: Add a light source
    bpy.ops.object.light_add(type='SUN', location=(10, -10, 10))
    light = bpy.context.object
    light.data.energy = 5

    # Step 4: Save the image
    save_image_file()


if __name__ == "__main__":
    main()
