from locust import HttpUser, task, between
from pathlib import Path

# Directories
IMAGE_DIR = Path(__file__).parent.parent / "images"

# Images
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


class APIUser(HttpUser):
    wait_time = between(0.5, 2)  # users wait between 0.5 and 2 seconds between tasks

    @task(5)
    def inference(self):
        """75% of calls (weight=5) go to single predict."""
        with open(IMAGE_DIR / "0001.jpg", "rb") as f:
            files = {"file": ("0001.jpg", f, "image/jpeg")}
            self.client.post("/api/v1/predict", files=files)

    @task(1)
    def batch_inference(self):
        files = []
        for path in BATCH_PATHS:
            f = open(path, "rb")
            files.append(("files", (path.name, f, "image/jpeg")))
        self.client.post("/api/v1/batch-predict", files=files)
        for _, (_, fh, _) in files:
            fh.close()

    @task(3)
    def sky_replacement(self):
        """45% of calls (weight=3) go to single sky replacement."""
        with open(IMAGE_DIR / "0001.jpg", "rb") as f:
            files = {"file": ("0001.jpg", f, "image/jpeg")}
            self.client.post("/api/v1/sky-replacement", files=files)

    @task(1)
    def batch_sky_replacement(self):
        files = []
        for path in BATCH_PATHS:
            f = open(path, "rb")
            files.append(("files", (path.name, f, "image/jpeg")))
        self.client.post("/api/v1/batch-sky-replacement", files=files)
        for _, (_, fh, _) in files:
            fh.close()
