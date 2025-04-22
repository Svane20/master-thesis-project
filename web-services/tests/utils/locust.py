import time
from pathlib import Path

from locust import FastHttpUser, between, events, task

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
            name="warm_latency",
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
            "sky_replacement_latency",
            "/api/v1/sky-replacement",
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
