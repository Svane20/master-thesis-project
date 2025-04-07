from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import io
from typing import List

from src.config import get_configuration
from src.middlewares import register_middlewares
from src.schemas import HealthResponse
from src.services.model_service import ModelService
from src.utils.logger import setup_logging

# Setup logging
setup_logging()

# Setup model pipeline
model_service = ModelService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        model_service.load_model()
    except Exception as e:
        raise e
    yield


version = "v1"
prefix = f"/api/{version}"
description = """
This API performs image matting and sky replacement for houses using a CPU.
The underlying model is a U-Net architecture with a ResNet-34 backbone.
This model was trained purely on synthetic data.
"""

app = FastAPI(
    title="U-Net ONNX API",
    description=description,
    version=version,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
    lifespan=lifespan
)

# Instrumentator setup for Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Middlewares
register_middlewares(app)


# Routes
@app.get(f"{prefix}/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.post(f"{prefix}/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    try:
        zip_bytes = await model_service.batch_predict(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=masks.zip"}
    )


@app.post(f"{prefix}/single-predict")
async def single_predict(file: UploadFile = File(...)):
    try:
        png_bytes = await model_service.single_predict(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")
    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=mask.png"}
    )


@app.post(f"{prefix}/sky-replacement")
async def sky_replacement(file: UploadFile = File(...)):
    try:
        zip_bytes = await model_service.sky_replacement(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=replacements.zip"}
    )


if __name__ == "__main__":
    configuration = get_configuration()

    uvicorn.run(app="src.main:app", host=configuration.HOST, port=configuration.PORT, reload=True)
