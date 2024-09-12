import json
import shutil
import os
import subprocess
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services.minio import upload_file_to_minio, download_file_from_minio, get_minio_files, delete_file_from_minio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check if Blender executable is available
blender_bin = shutil.which("blender")
if blender_bin is None:
    logging.error("Unable to find Blender executable")
    exit(1)
else:
    logging.info(f"Blender executable found: {blender_bin}")

BLENDER_EXECUTABLE = blender_bin
BLENDER_SCRIPT_NAME = "blender_operations.py"
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'blender', BLENDER_SCRIPT_NAME)

# Output directory for rendered images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

# Serve the output directory as a static directory
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


class RenderRequest(BaseModel):
    filename: str = 'test.png'


class RenderResponse(BaseModel):
    message: str
    image_path: Optional[str] = None


@app.get("/api/files")
async def get_files():
    response = get_minio_files()

    try:
        if response['status'] == 'success':
            return {"files": response['files']}
        else:
            raise HTTPException(status_code=400, detail=response['error'])

    except Exception as e:
        logger.error(f"Failed to get files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    response = upload_file_to_minio(file)

    try:
        if response['status'] == 'success':
            return {"message": "File uploaded successfully", "file_url": response["file_url"]}
        else:
            raise HTTPException(status_code=400, detail=response['error'])

    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/download/{filename}")
async def download_file(filename: str):
    try:
        file_data = download_file_from_minio(filename)

        if file_data['status'] == 'success':
            return StreamingResponse(
                file_data['file_data'],
                media_type="application/octet-stream",
                headers={'Content-Disposition': f'attachment; filename="{filename}"'}
            )
        else:
            raise HTTPException(status_code=400, detail=file_data['error'])

    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    response = delete_file_from_minio(filename)

    try:
        if response['status'] == 'success':
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail=response['error'])

    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/render", response_model=RenderResponse)
async def render_scene(request: Request, render_request: RenderRequest):
    filename = render_request.filename
    if not filename.lower().endswith(".png"):
        filename += ".png"

    output_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    params = {"objects": ["cube", "sphere"]}

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

    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Rendered image not found.")

    image_url = str(request.url_for("output", path=filename))

    return RenderResponse(message="Rendered image saved successfully.", image_path=image_url)
