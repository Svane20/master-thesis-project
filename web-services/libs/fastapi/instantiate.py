from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Tuple, Dict
from prometheus_fastapi_instrumentator import Instrumentator
import os
from typing import Optional

from libs.configuration.base import get_configuration
from libs.fastapi.consts import VERSION, LICENSE_INFO
from libs.fastapi.middlewares import register_middlewares
from libs.fastapi.settings import get_settings
from libs.services.base import BaseModelService
from libs.services.factory import get_model_service
from libs.logging import logger


def instantiate() -> Tuple[FastAPI, BaseModelService, Dict[str, str]]:
    """
    Instantiates a FastAPI application and a model service.

    Returns:
        Tuple[FastAPI, BaseModelService]: A tuple containing the FastAPI app, the model service and the project info.
    """
    # Get the settings
    settings = get_settings()
    deployment_type = "cuda" if settings.USE_GPU else "cpu"

    # Get the configuration
    configuration = get_configuration(settings.CONFIG_PATH)
    project_name = configuration.project_info.project_name
    model_type = configuration.project_info.model_type

    # Instantiate the model service
    model_service = get_model_service(settings, configuration)

    # Instantiate the FastAPI app
    fastapi_app = _instantiate_fastapi(
        model_service=model_service,
        project_name=project_name,
        model_type=model_type,
        deployment_type=deployment_type,
    )

    # Setup FastAPI configurations
    _setup_fastapi_configuration(fastapi_app)

    return fastapi_app, model_service, {
        "project_name": project_name,
        "model_type": model_type,
        "deployment_type": deployment_type,
    }


def get_workers() -> Optional[int]:
    """
    Read WORKERS from the environment. If unset or invalid, return None
    so Uvicorn will use its own default.
    """
    workers_env = os.getenv("WORKERS")
    if not workers_env:
        return None

    try:
        workers = int(workers_env)
        logger.info(f"Using {workers} workers")

        return workers
    except ValueError:
        logger.error(f"Invalid WORKERS value: {workers_env!r}")
        return None


def _setup_fastapi_configuration(fastapi_app: FastAPI) -> None:
    """
    Sets up the FastAPI application with necessary configurations.

    Args:
        fastapi_app (FastAPI): The FastAPI application instance.
    """
    # Setup Prometheus metrics
    Instrumentator().instrument(fastapi_app).expose(fastapi_app)

    # Register the middlewares
    register_middlewares(fastapi_app)


def _instantiate_fastapi(
        model_service: BaseModelService,
        project_name: str,
        model_type: str,
        deployment_type: str,
) -> FastAPI:
    """
    Instantiates a FastAPI instance with a custom lifespan context manager.

    Args:
        model_service (BaseModelService): The model service to be used.
        project_name (str): The name of the project.
        model_type (str): The type of model (e.g., ONNX, TorchScript).
        deployment_type (str): The type of deployment (CPU or GPU).

    Returns:
        FastAPI: An instance of FastAPI with a custom lifespan.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            model_service.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise Exception(f"Failed to load model: {e}")
        yield

    return FastAPI(
        title=_get_title(project_name, model_type, deployment_type),
        description=_get_description(project_name, deployment_type),
        version=VERSION,
        license_info=LICENSE_INFO,
        lifespan=lifespan,
        root_path=f"/{project_name}/{model_type}"
    )


def _get_title(project_name: str, model_type: str, deployment_type: str) -> str:
    """
    Generates a title for the FastAPI instance.

    Args:
        project_name (str): The name of the project.
        model_type (str): The type of model (e.g., ONNX, TorchScript).
        deployment_type (str): The type of deployment (CPU or GPU).

    Returns:
        str: A title string.
    """
    return f"{project_name.upper()} - {model_type.upper()} ({deployment_type}) API"


def _get_description(project_name: str, deployment_type: str) -> str:
    """
    Generates a description for the FastAPI instance.

    Args:
        project_name (str): The name of the project.
        deployment_type (str): The type of deployment (CPU or GPU).

    Returns:
        str: A description string.
    """
    deployment_type = "GPU" if deployment_type == "cuda" else "CPU"

    return f"""
    This API performs image matting and sky replacement for houses using a {deployment_type.upper()}.
    The underlying model is a U-Net architecture with a {project_name.upper()} backbone.
    This model was trained purely on synthetic data.
    """
