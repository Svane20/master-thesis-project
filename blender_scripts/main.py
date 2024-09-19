import bpy

from consts import CONFIG_PATH, OUTPUT_PATH, BLENDER_FILES_PATH
from rendering import setup_rendering
from settings import Settings, save_settings, load_settings, RenderType
from utils import save_blend_file


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    if bpy.data.objects.get("Cube"):
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()


clear_cube()

# Save the existing settings
settings = Settings()
save_settings(settings.model_dump(), CONFIG_PATH)

# Load settings
config = load_settings(CONFIG_PATH)
settings = Settings(**config)
terrain_settings = settings.terrain

setup_rendering(
    settings.render_settings,
    settings.camera_settings,
    render_object_index=True,
    world_size=int(terrain_settings.world_size)
)

save_blend_file(path=BLENDER_FILES_PATH)
