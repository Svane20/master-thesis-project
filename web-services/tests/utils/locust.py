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


def _fire(name: str, value: float, ts: float) -> None:
    """
    Record `value` in the response‑time column so it is visible
    in the Web UI and in the *_stats_history.csv file.
    """
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


def resource_monitor(environment, interval: int = 5):
    """
    Poll GPU utilization via nvidia-smi every interval_sec seconds
    and fire custom Locust metrics for GPU utilization and memory.
    """
    while _monitor_running:
        ts = time.time()
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

        except subprocess.CalledProcessError as e:
            _LOG.warning("nvidia-smi failed: %s", e)
        except ValueError as ve:
            _LOG.error("Parse error on nvidia‑smi output: %s", ve)
        except FileNotFoundError:
            _LOG.error("nvidia-smi not found – skip GPU metrics")
            return
        time.sleep(interval)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Start GPU resource monitor thread"""
    thread = threading.Thread(target=resource_monitor, args=(environment,), daemon=True)
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

    def post_request(self, name: str, url: str, files=None):
        """
        Helper to send a POST and record via events.request.fire.
        """
        start = time.perf_counter()
        try:
            response = self.client.post(url, files=files, timeout=120)
            response.raise_for_status()
            length = len(response.content) if response.content is not None else 0
            exception = None
        except Exception as e:
            length = 0
            exception = e
        latency = int((time.perf_counter() - start) * 1000)
        events.request.fire(
            request_type="POST",
            name=name,
            response_time=latency,
            response_length=length,
            exception=exception
        )
        return

    @task(3)
    def inference(self):
        self.post_request(
            name="inference_latency",
            url="/api/v1/predict",
            files={"file": ("0001.jpg", open(IMAGE_DIR / "0001.jpg", "rb"), "image/jpeg")}
        )

    @task(1)
    def batch_inference(self):
        files = [("files", (path.name, open(path, "rb"), "image/jpeg")) for path in BATCH_PATHS]
        self.post_request(
            name="batch_inference_latency",
            url="/api/v1/batch-predict",
            files=files
        )

    @task(5)
    def sky_replacement(self):
        self.post_request(
            name="sky_replacement_latency",
            url="/api/v1/sky-replacement",
            files={"file": ("0001.jpg", open(IMAGE_DIR / "0001.jpg", "rb"), "image/jpeg")}
        )

    @task(2)
    def batch_sky_replacement(self):
        files = [("files", (path.name, open(path, "rb"), "image/jpeg")) for path in BATCH_PATHS]
        self.post_request(
            name="batch_sky_replacement_latency",
            url="/api/v1/batch-sky-replacement",
            files=files
        )
