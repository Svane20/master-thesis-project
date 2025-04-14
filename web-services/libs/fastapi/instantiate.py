from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Tuple, Dict
from prometheus_fastapi_instrumentator import Instrumentator

from libs.fastapi.config import get_configuration, Settings
from libs.fastapi.consts import VERSION, LICENSE_INFO
from libs.fastapi.middlewares import register_middlewares
from libs.services.base import BaseModelService
from libs.services.factory import get_model_service
from libs.logging import logger


def instantiate(project_path: Path) -> Tuple[FastAPI, BaseModelService, Dict[str, str]]:
    """
    Instantiates a FastAPI application and a model service.

    Args:
        project_path (Path): The path to the project directory.

    Returns:
        Tuple[FastAPI, BaseModelService]: A tuple containing the FastAPI app, the model service and the project info.
    """
    # Get the defined configuration
    config = get_configuration()

    # Get the project name, model type, and deployment type
    project_name, model_type = _get_project_info(project_path)
    deployment_type = _get_deployment_type(config)

    # Instantiate the model service
    model_service = get_model_service(config, model_type)

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


def _get_project_info(project_path: Path) -> Tuple[str, str]:
    """
    Extracts project name and model type from the project path.

    Args:
        project_path (Path): The path to the project.

    Returns:
        Tuple[str, str]: A tuple containing the project name and model type.
    """
    parts = project_path.parts
    if len(parts) < 2:
        raise ValueError("Path must contain at least two parts for project and model type")

    # Extract project name and model type from the path
    project_name = parts[-2]
    model_type = parts[-1]

    return project_name, model_type


def _get_deployment_type(config: Settings) -> str:
    """
    Determines the deployment type based on the configuration.

    Args:
        config (Settings): The configuration settings.

    Returns:
        str: The deployment type (CPU or GPU).
    """
    return "cuda" if config.USE_GPU else "cpu"
