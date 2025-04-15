from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, List
import io

from libs.fastapi.consts import API_PREFIX
from libs.fastapi.schemas import HealthResponse, InfoResponse
from libs.services.base import BaseModelService


def register_routes(app: FastAPI, model_service: BaseModelService, project_info: Dict[str, str]) -> None:
    @app.post(
        path=f"{API_PREFIX}/sky-replacement",
        tags=["Sky Replacement"],
        responses={
            200: {
                "content": {"image/png": {}},
                "description": "The replaced image in PNG format.",
            },
            500: {"description": "Inference failed"},
        },
    )
    async def sky_replacement(file: UploadFile = File(...)):
        try:
            png_bytes = await model_service.sky_replacement(file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to perform sky replacement: {e}")

        return StreamingResponse(
            io.BytesIO(png_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=replaced.png"}
        )

    @app.post(
        path=f"{API_PREFIX}/sky-replacement-extra",
        tags=["Sky Replacement"],
        responses={
            200: {
                "content": {"application/zip": {}},
                "description": "A ZIP archive containing the predicted alpha matte, estimated foreground and replaced image.",
            },
            500: {"description": "Inference failed"},
        },
    )
    async def sky_replacement(file: UploadFile = File(...)):
        try:
            zip_bytes = await model_service.sky_replacement(file, extra=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to perform sky replacement: {e}")

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=replacements.zip"}
        )

    @app.post(
        path=f"{API_PREFIX}/single-predict",
        tags=["Inference"],
        responses={
            200: {
                "content": {"image/png": {}},
                "description": "The predicted alpha matte in PNG format.",
            },
            500: {"description": "Inference failed"},
        },
    )
    async def single_predict(file: UploadFile = File(...)):
        try:
            response_bytes = await model_service.single_predict(file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")

        return StreamingResponse(
            io.BytesIO(response_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=alpha.png",
                "Content-Length": str(len(response_bytes))
            }
        )

    @app.post(
        path=f"{API_PREFIX}/batch-predict",
        tags=["Inference"],
        responses={
            200: {
                "content": {"application/zip": {}},
                "description": "A ZIP archive containing the predicted alpha mattes.",
            },
            500: {"description": "Inference failed"},
        },
    )
    async def batch_predict(files: List[UploadFile] = File(...)):
        try:
            zip_bytes = await model_service.batch_predict(files)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=alphas.zip",
                "Content-Length": str(len(zip_bytes))
            }
        )

    @app.get(path=f"{API_PREFIX}/health", response_model=HealthResponse, tags=["Info"])
    async def health():
        return {"status": "ok"}

    @app.get(path=f"{API_PREFIX}/info", response_model=InfoResponse, tags=["Info"])
    async def info():
        return project_info
