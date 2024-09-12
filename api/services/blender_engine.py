import subprocess
import json
import shutil
import logging
import os

from fastapi import HTTPException

from services.minio import upload_file_to_minio_from_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global Blender configuration
BLENDER_EXECUTABLE = shutil.which("blender")
BLENDER_SCRIPT_DIR = os.path.dirname(__file__)
BLENDER_SCRIPT = os.path.join(BLENDER_SCRIPT_DIR, 'blender', 'blender_operations.py')

# MINIO Configuration
MINIO_BUCKET = "rendered-images"

if BLENDER_EXECUTABLE is None:
    logger.error("Blender executable not found")
    raise SystemExit("Blender executable not found. Ensure Blender is installed and accessible in the system PATH.")
else:
    logger.info(f"Blender executable found at: {BLENDER_EXECUTABLE}")


async def start_rendering_process(output_path: str, params: dict, filename: str):
    """
    Function that handles rendering the image and uploading it to MinIO in the background.
    """
    try:
        await run_blender(output_path, params)

        minio_key = filename
        upload_result = upload_file_to_minio_from_path(output_path, minio_key)

        if upload_result['status'] == 'success':
            logger.info(f"Rendered image uploaded to MinIO at {upload_result['file_url']}")
        else:
            logger.error(f"Failed to upload rendered image to MinIO: {upload_result['error']}")

        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Local file {output_path} removed after upload.")

    except Exception as e:
        logger.error(f"An error occurred while processing the render job: {e}")


async def run_blender(output_path: str, params: dict):
    """
    Runs Blender synchronously with the provided output path and parameters.

    Args:
        output_path (str): The path to save the rendered output.
        params (dict): Parameters to pass to the Blender script.

    Raises:
        HTTPException: If the Blender process fails or encounters an error.
    """
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        logger.info(f"Blender rendering completed successfully. Output: {result.stdout}")
        return result.stdout

    except subprocess.CalledProcessError as e:
        logger.error(f"Blender rendering failed: {e.stderr}")
        raise HTTPException(status_code=500, detail="Blender rendering failed.")
    except Exception as e:
        logger.exception("An error occurred while running the Blender process")
        raise HTTPException(status_code=500, detail="Blender rendering failed due to an internal error.")
