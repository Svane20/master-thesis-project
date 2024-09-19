import bpy

from consts import OUTPUT_PATH
from settings import RenderSettings, CameraSettings


def setup_rendering(
        render_settings: RenderSettings,
        camera_settings: CameraSettings,
        output_dir: str = OUTPUT_PATH,
        render_image: bool = True,
        render_object_index: bool = True,
        render_material_index: bool = False,
        render_depth: bool = False,
        render_mist: bool = False,
        render_normal: bool = False,
        world_size: int = 100,
):
    scene = bpy.context.scene
    render = scene.render
    cycles = scene.cycles

    # Render settings
    render.engine = render_settings.render.value
    render.resolution_x = int(camera_settings.image_width)
    render.resolution_y = int(camera_settings.image_height)

    # Cycles settings
    cycles.camera_cull_margin = render_settings.camera_cull_margin
    cycles.distance_cull_margin = render_settings.distance_cull_margin
    cycles.use_camera_cull = True
    cycles.use_distance_cull = True



def setup_cuda(
        render_settings: RenderSettings,
):
    scene = bpy.data.scenes["Scene"]
    render = bpy.context.scene.render


