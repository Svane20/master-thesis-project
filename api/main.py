import json
import shutil
import subprocess
import os

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Output directory for rendered images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve the output directory as a static directory
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Check if Blender executable is available
blender_bin = shutil.which("blender")
if blender_bin is None:
    logging.error("Unable to find Blender!")
    exit(1)

BLENDER_EXECUTABLE = blender_bin
BLENDER_SCRIPT_NAME = "blender_operations.py"
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'blender', BLENDER_SCRIPT_NAME)

logging.info(f"Using Blender executable: {BLENDER_EXECUTABLE}")


class RenderRequest(BaseModel):
    filename: str = 'test.png'


class RenderResponse(BaseModel):
    message: str
    image_path: str = None


@app.post("/api/render/", response_model=RenderResponse)
async def render_scene(request: Request, render_request: RenderRequest):
    filename = render_request.filename

    # Ensure the filename ends with .png
    if not filename.lower().endswith(".png"):
        filename += ".png"

    output_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))

    params = {
        "objects": ["cube", "sphere"],
    }

    try:
        result = subprocess.run(
            [
                BLENDER_EXECUTABLE,
                "--debug-all",
                "--background",
                "--python",
                BLENDER_SCRIPT,
                "--",
                output_path,
                json.dumps(params)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        logging.info(f"Blender output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Blender rendering failed: {e.stderr}")
        raise HTTPException(status_code=500, detail="Blender rendering failed.")

    # Check if the image was created
    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Rendered image not found.")

    # Generate URL for the rendered image using the Request object
    image_url = str(request.url_for("output", path=filename))

    return RenderResponse(message="Rendered image saved successfully.", image_path=image_url)


@app.get("/api/images")
async def get_images(request: Request):
    images = []
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".png"):
            # Generate full URLs for each image
            image_url = request.url_for("output", path=file)
            images.append(image_url)
    return images


@app.get("/api/images/{filename}")
async def get_image(filename: str, request: Request):
    if not filename.endswith(".png"):
        filename += ".png"

    image_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    return request.url_for("output", path=filename)
