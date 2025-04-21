import threading
import time
from pathlib import Path
import os

import requests
import uvicorn

from libs.fastapi.settings import reset_settings_cache
from tests.utils.configuration import get_performance_custom_path
from tests.utils.factory import get_create_app_func

# Project and Model Type
current_directory = Path(__file__).parent
project_name = current_directory.parent.name
model_type = current_directory.name
use_gpu = True


def start_uvicorn(app, port, workers=None, timeout: float = 10.0, interval: float = 0.1):
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": "0.0.0.0",
            "port": port,
            "workers": workers,
        },
        daemon=True,
    )
    thread.start()

    # Do not return until the server is ready
    # 2) Poll the health endpoint until it returns 200 or we timeout
    start = time.monotonic()
    url = f"http://localhost:{port}/api/v1/live"
    while True:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return  # healthy!
        except requests.RequestException:
            # server not up yet or other network hiccup
            pass

        if time.monotonic() - start > timeout:
            raise RuntimeError(f"Server did not respond 200 at {url} within {timeout}s")
        time.sleep(interval)


cfg_path = get_performance_custom_path(project_name, model_type)
os.environ["CONFIG_PATH"] = cfg_path
os.environ["USE_GPU"] = "true" if use_gpu else "false"
os.environ["MAX_BATCH_SIZE"] = "8"

reset_settings_cache()

create_app = get_create_app_func(project_name, model_type)
app = create_app()

# Start the Uvicorn server in a separate thread
start_uvicorn(app, port=8006)

print(f"Here")
