import json
import shutil
import subprocess
import os

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Check if Blender executable is available
blender_bin = shutil.which("blender")
if blender_bin is None:
    logging.error("Unable to find Blender!")
    exit(1)

BLENDER_EXECUTABLE = blender_bin
BLENDER_SCRIPT_NAME = "blender_operations.py"
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'blender', BLENDER_SCRIPT_NAME)

logging.info(f"Using Blender executable: {BLENDER_EXECUTABLE}")

# Output directory for rendered images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class RenderRequest(BaseModel):
    filename: str = 'test.png'


class RenderResponse(BaseModel):
    message: str
    image_path: str = None


@app.post("/api/render/", response_model=RenderResponse)
async def render_scene(request: RenderRequest):
    filename = request.filename

    # Ensure the filename ends with .png
    if not filename.lower().endswith(".png"):
        filename += ".png"

    output_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))

    # @todo: Add parameters to the Blender script
    params = {
        "objects": ["cube", "sphere"],
    }

    try:
        result = subprocess.run(
            [
                BLENDER_EXECUTABLE,
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

    return RenderResponse(message="Rendered image saved successfully.", image_path=output_path)
