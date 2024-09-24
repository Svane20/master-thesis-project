import bpy
from mathutils import Vector

from math import radians

from consts import Constants
from rendering.camera import update_camera_position
from main import setup
from rendering.light import create_light, LightType
from utils import save_blend_file, open_blend_file_in_blender, save_image_file, run_blender_file


def _remove_default_objects() -> None:
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def _create_material(name: str, color: tuple[float, float, float], alpha: float = 1.0) -> bpy.types.Material:
    """Create a new material with the given color."""
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = (*color, alpha)  # RGB + Alpha
    return mat


def _add_image_texture_to_plane(image_name: str) -> None:
    """Adds an image texture to the plane."""
    # Select the plane
    plane = bpy.context.active_object

    # Create a new material
    mat = bpy.data.materials.new(name="GrassMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')

    # Load image
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(str(Constants.Directory.ASSETS_TEXTURES_GRASS_DIR / image_name))
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Assign material to the plane
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)


def _add_house_elements() -> None:
    # Add base
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 1))
    house_base = bpy.context.object
    house_base.name = "Base"

    # Add roof
    bpy.ops.mesh.primitive_cone_add(
        vertices=4,
        radius1=1.5,
        depth=1,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 2.5),
    )
    roof = bpy.context.object
    roof.name = "Roof"
    roof.rotation_euler = (0, 0, radians(45))

    # Add door
    bpy.ops.mesh.primitive_cube_add(size=0.5, enter_editmode=False, align='WORLD', location=(0, -1.01, 0.25))
    door = bpy.context.object
    door.name = "Door"
    door.scale = (0.3, 0.05, 0.6)

    # Add red material to the door
    red_material = _create_material("RedMaterial", (1.0, 0, 0))
    door.data.materials.append(red_material)

    # Add windows
    bpy.ops.mesh.primitive_cube_add(size=0.6, enter_editmode=False, align='WORLD', location=(-0.6, -1.01, 0.9))
    window_left = bpy.context.object
    window_left.name = "Left Window"
    window_left.scale = (0.3, 0.05, 0.3)

    bpy.ops.mesh.primitive_cube_add(size=0.6, enter_editmode=False, align='WORLD', location=(0.6, -1.01, 0.9))
    window_right = bpy.context.object
    window_right.name = "Right Window"
    window_right.scale = (0.3, 0.05, 0.3)

    # Add blue material to the windows
    blue_material = _create_material("BlueMaterial", (0, 0, 1.0))

    for window in [window_left, window_right]:
        window.data.materials.append(blue_material)


def main() -> None:
    setup(output_name="plain")

    _remove_default_objects()

    # Add a plane to the scene and add a grass texture to it
    bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    _add_image_texture_to_plane("brown_mud_leaves_01_diff_1k.jpg")

    # Add house elements to the scene
    _add_house_elements()

    # Update camera position
    bpy.context.scene.camera = update_camera_position(
        location=Vector((0.0, -7.0, 2.0)),
        rotation=Vector((radians(80), 0.0, 0.0))
    )

    # Add a light to the scene
    create_light(
        light_name="Sun",
        light_type=LightType.SUN,
        energy=5.0,
    )

    # Save the blend file and open it in Blender
    # save_blend_file(path=Constants.Directory.BLENDER_FILES_PATH)
    # open_blend_file_in_blender(blender_file=Constants.Directory.BLENDER_FILES_PATH)
    # run_blender_file(Constants.Directory.BLENDER_FILES_PATH)

    # Save the blend file as a PNG image
    save_image_file()


if __name__ == "__main__":
    main()
