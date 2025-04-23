from fastapi import FastAPI
import uvicorn

from libs.fastapi.instantiate import instantiate, get_workers
from libs.fastapi.routes import register_routes


def create_app() -> FastAPI:
    app, model_service, project_info = instantiate()
    register_routes(app, model_service, project_info)
    return app


if __name__ == "__main__":
    uvicorn.run(app="main:create_app", factory=True, workers=get_workers(), host="0.0.0.0", port=8001, reload=True)
