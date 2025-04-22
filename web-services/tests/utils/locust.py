import random
import time
import threading
import subprocess
from pathlib import Path
import logging

from locust import FastHttpUser, task, between, events

# Directories
IMAGE_DIR = Path(__file__).parent.parent / "images"

# Batch image list
BATCH_PATHS = [
    IMAGE_DIR / "0001.jpg",
    IMAGE_DIR / "0055.jpg",
    IMAGE_DIR / "0086.jpg",
    IMAGE_DIR / "0211.jpg",
    IMAGE_DIR / "1901.jpg",
    IMAGE_DIR / "2022.jpg",
    IMAGE_DIR / "2041.jpg",
    IMAGE_DIR / "10406.jpg",
]

# Flag for monitor thread to exit cleanly
_monitor_running = True

# Logging
_LOG = logging.getLogger("gpu-monitor")


def _fire(name: str, value: float, ts: int) -> None:
    """Emit a synthetic request so the value shows up in Locust stats."""
    events.request.fire(
        request_type="GPU",
        name=name,
        response_time=value,
        response_length=0,
        exception=None,
        start_time=ts,
        url="nvidia-smi",
        context={},
        response=None,
    )


def _resource_monitor(environment, interval: int = 5) -> None:
    """
    Poll GPU utilization via nvidia-smi every interval_sec seconds
    and fire custom Locust metrics for GPU utilization and memory.
    """
    while _monitor_running:
        ts = int(time.time())
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )
            for line in out.strip().splitlines():
                gpu_id, util, mem = (x.strip() for x in line.split(","))
                _fire(f"gpu{gpu_id}_util_%", float(util), ts)
                _fire(f"gpu{gpu_id}_mem_MiB", float(mem), ts)

        except FileNotFoundError:
            _LOG.error("nvidia-smi not found – skipping GPU metrics")
            return
        except subprocess.CalledProcessError as e:
            _LOG.warning("nvidia-smi failed: %s", e)
        except ValueError as ve:
            _LOG.error("Parse error on nvidia‑smi output: %s", ve)

        time.sleep(interval)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Start GPU resource monitor thread"""
    thread = threading.Thread(target=_resource_monitor, args=(environment,), daemon=True)
    thread.start()


@events.quitting.add_listener
def on_locust_quit(environment, **kwargs):
    global _monitor_running
    _monitor_running = False


class APIUser(FastHttpUser):
    """
    Locust user for hitting FastAPI via NGINX load‑balanced path.
    Uses try/except around requests, custom timeouts, and records failures.
    """
    wait_time = between(0.5, 2)

    def _post(self, url: str, files=None):
        self.client.post(url, files=files, timeout=120)

    @task(3)
    def inference(self):
        path = random.choice(BATCH_PATHS)
        file = {"file": (path.name, open(path, "rb"), "image/jpeg")}
        self._post("/api/v1/predict", files=file)

    @task(1)
    def batch_inference(self):
        paths = random.sample(BATCH_PATHS, k=len(BATCH_PATHS))
        files = [("files", (p.name, open(p, "rb"), "image/jpeg")) for p in paths]
        self._post("/api/v1/batch-predict", files=files)

    @task(5)
    def sky_replacement(self):
        path = random.choice(BATCH_PATHS)
        file = {"file": (path.name, open(path, "rb"), "image/jpeg")}
        self._post("/api/v1/sky-replacement", files=file)

    @task(2)
    def batch_sky_replacement(self):
        paths = random.sample(BATCH_PATHS, k=len(BATCH_PATHS))
        files = [("files", (p.name, open(p, "rb"), "image/jpeg")) for p in paths]
        self._post("/api/v1/batch-sky-replacement", files=files)
