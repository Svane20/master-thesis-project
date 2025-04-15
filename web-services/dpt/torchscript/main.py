import uvicorn

from libs.fastapi.instantiate import instantiate
from libs.fastapi.routes import register_routes

# Instantiate the FastAPI app and model service
app, model_service, project_info = instantiate()

# Register the routes
register_routes(app, model_service, project_info)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8008, reload=True)
