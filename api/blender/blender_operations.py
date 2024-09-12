import bpy
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clear_scene():
    """Remove all objects from the current scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_scene(params):
    """Set up the scene with objects, camera, and lighting based on input parameters."""
    clear_scene()

    logging.info(f"Setting up scene with objects: {params.get('objects', [])}")

    # Add objects based on the object parameter
    for obj_type in params.get('objects', []):
        if obj_type == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
        elif obj_type == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(3, 0, 0))
        else:
            logger.exception("Unsupported object type: %s", obj_type)

    # Add lighting
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 1000
    light.name = "PointLight"

    # Add a camera
    bpy.ops.object.camera_add(location=(0, -10, 5))  # Adjust camera location
    camera = bpy.context.active_object
    camera.name = "Camera"

    # Adjust the camera rotation to point towards the center of the objects
    camera.rotation_euler = (1.2, 0, 0)  # Rotate to face objects

    # Set the camera as the active camera
    bpy.context.scene.camera = camera

def configure_render(output_path):
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path

    scene.cycles.device = 'CPU'  # Force rendering on the CPU

    # Ensure denoising is disabled
    scene.cycles.use_denoising = False

def render_image():
    """Render the image and save it to the specified path."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Extract arguments after "--"
    else:
        argv = []

    output_path = argv[0] if len(argv) > 0 else "output.png"

    logger.info("Output path: %s", output_path)

    if len(argv) < 1:
        logger.info("No parameters provided. Using default values.")
        params = {}
    else:
        params = json.loads(argv[1])

    # Now set up the scene and render the image based on the parsed parameters
    setup_scene(params)
    configure_render(output_path)
    bpy.ops.render.render(write_still=True)
    logger.info("Rendered image saved to %s", output_path)

render_image()
