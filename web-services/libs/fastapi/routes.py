from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict
import io

from libs.fastapi.consts import API_PREFIX
from libs.fastapi.schemas import HealthResponse, InfoResponse
from libs.services.base import BaseModelService


def register_routes(app: FastAPI, model_service: BaseModelService, project_info: Dict[str, str]) -> None:
    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
    async def health():
        return {"status": "ok"}

    @app.get(f"{API_PREFIX}/info", response_model=InfoResponse)
    async def info():
        return project_info

    @app.post(f"{API_PREFIX}/single-predict")
    async def single_predict(file: UploadFile = File(...)):
        try:
            response_bytes = await model_service.single_predict(file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")

        return StreamingResponse(
            io.BytesIO(response_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=alpha.png"}
        )
