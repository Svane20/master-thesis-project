import os
import threading
import time

import requests
from uvicorn import Config, Server

from libs.fastapi.settings import reset_settings_cache
from tests.utils.configuration import get_performance_custom_path
from tests.utils.factory import get_create_app_func


def setup_api_instance(project_name: str, model_type: str, use_gpu: bool, workers: int = None) -> Server:
    # Create the custom config path
    cfg_path = get_performance_custom_path(project_name, model_type)

    # Override the environment variables so the app loads the custom config
    os.environ["CONFIG_PATH"] = cfg_path
    os.environ["USE_GPU"] = "true" if use_gpu else "false"
    os.environ["MAX_BATCH_SIZE"] = "8"

    # Reset the settings cache to ensure the new environment variable is picked up
    reset_settings_cache()

    # Create the app
    create_app = get_create_app_func(project_name, model_type)
    app = create_app()

    return _start_uvicorn(app, workers)


def _start_uvicorn(app, workers=None) -> Server:
    """
    Start the Uvicorn server in a separate thread.

    Args:
        app: The FastAPI application to run.
        workers (int, optional): Number of worker processes. Defaults to None.
    """
    cfg = Config(app=app, host="0.0.0.0", port=8000, workers=workers, reload=False)
    server = Server(cfg)

    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-server")
    thread.start()
    print(f"INFO:     Uvicorn thread started: {thread.name}")

    _wait_for_server_ready()

    return server


def _wait_for_server_ready(timeout: float = 10.0, interval: float = 0.1) -> None:
    """
    Wait for the server to be ready.

    Args:
        timeout (float): The maximum time to wait for the server to be ready.
        interval (float): The time to wait between checks.
    """
    url = f"http://localhost:8000/api/v1/live"

    start = time.monotonic()
    while True:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                print("INFO:     Server is ready")
                return
        except requests.RequestException:
            pass

        if time.monotonic() - start > timeout:
            raise RuntimeError(f"Server did not respond 200 at {url} within {timeout}s")
        time.sleep(interval)
