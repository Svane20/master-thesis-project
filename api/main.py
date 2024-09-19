import logging
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
import requests

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.blender_engine import start_rendering_process
from services.minio import upload_file_to_minio, download_file_from_minio, get_minio_files, delete_file_from_minio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if os.getenv('ENVIRONMENT') == 'PROD':
    load_dotenv('.env.prod')
else:
    load_dotenv('.env.dev')

FLAMENCO_MANAGER_URL = os.getenv('FLAMENCO_MANAGER_URL')

# Output directory for rendered images
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Serve the output directory as a static directory
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# Model to define the job submission payload
class JobSubmission(BaseModel):
    blend_file: str
    frame_start: int
    frame_end: int
    output_path: str
    project: str
    user_email: str
    user_name: str
    chunk_size: int = 3
    format: str = "PNG"
    fps: int = 24
    has_previews: bool = False
    image_file_extension: str = ".png"
    priority: int = 50
    submitter_platform: str = "linux"
    job_type: str = "simple-blender-render"

class RenderRequest(BaseModel):
    filename: str = 'test.png'


class RenderResponse(BaseModel):
    message: str


@app.post("/api/submit-job")
def submit_job(job: JobSubmission):
    # Prepare the payload for Flamenco Manager
    payload = {
        "metadata": {
            "project": job.project,
            "user.email": job.user_email,
            "user.name": job.user_name
        },
        "name": "Blender Render Job",
        "priority": job.priority,
        "settings": {
            "blendfile": job.blend_file,
            "chunk_size": job.chunk_size,
            "format": job.format,
            "fps": job.fps,
            "frames": f"{job.frame_start}-{job.frame_end}",
            "has_previews": job.has_previews,
            "image_file_extension": job.image_file_extension,
            "render_output_path": job.output_path,
            "render_output_root": "/var/flamenco/output"  # Adjust based on your shared storage path
        },
        "submitter_platform": job.submitter_platform,
        "type": job.job_type
    }

    try:
        response = requests.post(f"{FLAMENCO_MANAGER_URL}/api/v3/jobs", json=payload)
        response.raise_for_status()
        return {"status": "success", "payload": response.json()}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {e}")


@app.get("/api/files")
async def get_files():
    try:
        response = get_minio_files()
        return handle_minio_response(response)
    except Exception as e:
        logger.error(f"Failed to get files: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        response = upload_file_to_minio(file)
        return handle_minio_response(response, success_message={"message": "File uploaded successfully",
                                                                "file_url": response["file_url"]})
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@app.get("/api/files/download/{filename}")
async def download_file(filename: str):
    try:
        file_data = download_file_from_minio(filename)
        if file_data['status'] == 'success':
            # Return StreamingResponse directly instead of passing it through handle_minio_response
            return StreamingResponse(
                file_data['file_data'],
                media_type="application/octet-stream",
                headers={'Content-Disposition': f'attachment; filename="{filename}"'}
            )
        else:
            raise HTTPException(status_code=400, detail=file_data['error'])
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        raise HTTPException(status_code=500, detail="File download failed")


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    try:
        response = delete_file_from_minio(filename)
        return handle_minio_response(response, success_message={"message": "File deleted successfully"})
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        raise HTTPException(status_code=500, detail="File deletion failed")


@app.post("/api/render", response_model=RenderResponse)
async def render_scene(render_request: RenderRequest, background_tasks: BackgroundTasks):
    filename = render_request.filename
    output_path = str(OUTPUT_DIR / filename)
    params = {"objects": ["cube", "sphere"]}

    background_tasks.add_task(start_rendering_process, output_path, params, filename)

    return RenderResponse(message="Rendering process started")


def handle_minio_response(response, success_message: Optional[dict] = None):
    """
    Handles the response from MinIO operations.
    If the operation is successful, returns the provided success_message.
    If the operation fails, raises an HTTPException with the error message.
    """
    if response['status'] == 'success':
        return success_message if success_message else response
    else:
        logger.error(f"MinIO operation failed: {response.get('error', 'Unknown error')}")
        raise HTTPException(status_code=400, detail=response.get('error', 'Unknown error'))
