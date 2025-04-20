import time
import threading
import subprocess
from pathlib import Path

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


def resource_monitor(environment, interval_sec: int = 5):
    """
    Poll GPU utilization via nvidia-smi every interval_sec seconds
    and fire custom Locust metrics for GPU utilization and memory.
    """
    while _monitor_running:
        try:
            output = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"
            ], encoding="utf-8")
            util_str, mem_str = output.strip().split(", ")
            util = float(util_str)
            mem = float(mem_str)

            events.request.fire(
                request_type="GPU",
                name="gpu_utilization_percent",
                response_time=0,
                response_length=int(util),
                exception=None
            )
            events.request.fire(
                request_type="GPU",
                name="gpu_memory_used_mib",
                response_time=0,
                response_length=int(mem),
                exception=None
            )
        except Exception:
            # ignore errors
            pass
        time.sleep(interval_sec)


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
    host = "http://localhost"

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

    def on_start(self):
        # cold start
        self.post_request("cold_start_latency", "/swin/onnx/api/v1/predict",
                          files={"file": ("0001.jpg", open(IMAGE_DIR / "0001.jpg", "rb"), "image/jpeg")})

    @task(3)
    def inference(self):
        self.post_request("warm_latency", "/swin/onnx/api/v1/predict",
                          files={"file": ("0001.jpg", open(IMAGE_DIR / "0001.jpg", "rb"), "image/jpeg")})

    @task(1)
    def batch_inference(self):
        files = [("files", (path.name, open(path, "rb"), "image/jpeg")) for path in BATCH_PATHS]
        self.post_request("batch_inference_latency", "/swin/onnx/api/v1/batch-predict", files=files)

    @task(5)
    def sky_replacement(self):
        self.post_request("sky_replacement_latency", "/swin/onnx/api/v1/sky-replacement",
                          files={"file": ("0001.jpg", open(IMAGE_DIR / "0001.jpg", "rb"), "image/jpeg")})

    @task(2)
    def batch_sky_replacement(self):
        files = [("files", (path.name, open(path, "rb"), "image/jpeg")) for path in BATCH_PATHS]
        self.post_request("batch_sky_replacement_latency", "/swin/onnx/api/v1/batch-sky-replacement", files=files)
