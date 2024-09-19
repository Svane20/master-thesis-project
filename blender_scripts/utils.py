import bpy

def save_blend_file(path: str) -> None:
    """Saves the current Blender scene as a .blend file."""
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"File saved successfully to {path}")

def save_image_file() -> None:
    """Saves the current Blender scene as a .png file."""
    bpy.ops.render.render(write_still=True)