from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from asgi_lifespan import LifespanManager
import asyncio
import time

from libs.fastapi.settings import reset_settings_cache
from tests.utils.factory import get_create_app_func


def get_test_client(
        request,
        custom_config_path,
        monkeypatch,
        project_name: str,
        model_type: str
):
    use_gpu = request.param if hasattr(request, "param") else False

    # Override the environment variables so the app loads the custom config
    monkeypatch.setenv("USE_GPU", "true" if use_gpu else "false")
    monkeypatch.setenv("CONFIG_PATH", custom_config_path)
    monkeypatch.setenv("MAX_BATCH_SIZE", "8")

    # Reset the settings cache to ensure the new environment variable is picked up
    reset_settings_cache()

    # Create the app
    app = get_create_app_func(project_name, model_type)()

    with TestClient(app) as c:
        # Create a variable to check which hardware is being used
        c.use_gpu = use_gpu

        yield c


async def get_performance_client(
        request,
        custom_config_path,
        monkeypatch,
        project_name: str,
        model_type: str
):
    use_gpu = request.param

    # Override the environment variables so the app loads the custom config
    monkeypatch.setenv("USE_GPU", "true" if use_gpu else "false")
    monkeypatch.setenv("CONFIG_PATH", custom_config_path)
    monkeypatch.setenv("MAX_BATCH_SIZE", "8")

    # Reset the settings cache to ensure the new environment variable is picked up
    reset_settings_cache()

    # Create the app
    app = get_create_app_func(project_name, model_type)()

    # Create the transport and client
    transport = ASGITransport(app=app)
    async with LifespanManager(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            ac.use_gpu = request.param
            await _wait_for_health(ac)
            yield ac


async def _wait_for_health(client, path="api/v1/live", timeout=10.0, interval=0.1):
    """
    Polls GET /live until we get 200 or timeout.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        r = await client.get(path)
        if r.status_code == 200:
            return
        await asyncio.sleep(interval)
    raise RuntimeError(f"Service did not respond healthy within {timeout}s")
