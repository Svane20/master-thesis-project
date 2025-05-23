from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from typing import Dict, List
import io

from libs.fastapi.consts import API_PREFIX
from libs.fastapi.schemas import HealthResponse, InfoResponse, LivenessResponse
from libs.services.base import BaseModelService


def register_routes(app: FastAPI, model_service: BaseModelService, project_info: Dict[str, str]) -> None:
    @app.post(
        path=f"{API_PREFIX}/sky-replacement",
        tags=["Sky Replacement"],
        responses={
            status.HTTP_200_OK: {
                "content": {"image/png": {}},
                "description": "The replaced image in PNG format.",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Inference failed"},
        },
    )
    async def sky_replacement(
            file: UploadFile = File(...),
            extra: bool = Query(
                False,
                title="Extra Output",
                description="If true, include alpha mattes and foregrounds in the ZIP",
            ),
    ):
        try:
            bytes = await model_service.sky_replacement(file, extra=extra)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to perform sky replacement: {e}")

        if extra:
            return StreamingResponse(
                io.BytesIO(bytes),
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=replacement.zip",
                    "Content-Length": str(len(bytes))
                }
            )

        return StreamingResponse(
            io.BytesIO(bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=replaced.png"}
        )

    @app.post(
        path=f"{API_PREFIX}/batch-sky-replacement",
        tags=["Sky Replacement"],
        responses={
            status.HTTP_200_OK: {
                "content": {"application/zip": {}},
                "description": "A ZIP archive containing the replaced images.",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Inference failed"},
        },
    )
    async def batch_sky_replacement(
            files: List[UploadFile] = File(...),
            extra: bool = Query(
                False,
                title="Extra Output",
                description="If true, include alpha mattes and foregrounds in the ZIP",
            ),
    ):
        try:
            zip_bytes = await model_service.batch_sky_replacement(files, extra=extra)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to perform sky replacement: {e}"
            )

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=replacements.zip",
                "Content-Length": str(len(zip_bytes))
            }
        )

    @app.post(
        path=f"{API_PREFIX}/predict",
        tags=["Inference"],
        responses={
            status.HTTP_200_OK: {
                "content": {"image/png": {}},
                "description": "The predicted alpha matte in PNG format.",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Inference failed"},
        },
    )
    async def single_predict(file: UploadFile = File(...)):
        try:
            response_bytes = await model_service.single_predict(file)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to predict: {e}"
            )

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
            status.HTTP_200_OK: {
                "content": {"application/zip": {}},
                "description": "A ZIP archive containing the predicted alpha mattes.",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Inference failed"},
        },
    )
    async def batch_predict(files: List[UploadFile] = File(...)):
        try:
            zip_bytes = await model_service.batch_predict(files)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to predict: {e}"
            )

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

    @app.get(
        f"{API_PREFIX}/live",
        response_model=LivenessResponse,
        tags=["Info"],
        responses={
            status.HTTP_200_OK: {"description": "Model is loaded and ready"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not yet loaded"}
        },
    )
    async def live():
        """
        Readiness probe: returns 200 only once the model is fully loaded.
        """
        if not model_service.is_ready():
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not ready")
        return {"status": "ready"}

    @app.get(path=f"{API_PREFIX}/info", response_model=InfoResponse, tags=["Info"])
    async def info():
        return project_info
