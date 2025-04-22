import random
from pathlib import Path
from typing import List

from locust import FastHttpUser, task, between, LoadTestShape

# Directories
IMAGE_DIR = Path(__file__).parent.parent / "images"
BATCH: List[Path] = [p for p in IMAGE_DIR.glob("*.jpg")][:8]


class StepLoadShape(LoadTestShape):
    steps: List[int] = [5, 10, 20, 40]
    step_duration: int = 120

    def tick(self):
        run_time = self.get_run_time()
        step_index = int(run_time // self.step_duration)
        if step_index < len(self.steps):
            users = self.steps[step_index]
            return users, max(1, users // 10)
        return None


class APIUser(FastHttpUser):
    wait_time = between(0.5, 2)

    def _post(self, url: str, files):
        self.client.post(url, files=files, timeout=120)

    @task(3)
    def inference(self):
        path = random.choice(BATCH)
        file = {"file": (path.name, path.read_bytes(), "image/jpeg")}
        self._post("/api/v1/predict", file)

    @task(5)
    def sky_replacement(self):
        path = random.choice(BATCH)
        file = {"file": (path.name, path.read_bytes(), "image/jpeg")}
        self._post("/api/v1/sky-replacement", file)
