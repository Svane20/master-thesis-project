import random
from pathlib import Path
from typing import List
import os

from locust import FastHttpUser, task, between, LoadTestShape

# Directories
IMAGE_DIR = Path(__file__).parent.parent / "images"
BATCH: List[Path] = [p for p in IMAGE_DIR.glob("*.jpg")][:8]

_default_steps = [5, 10, 20, 30, 40, 50]
_steps_env = os.getenv("MAX_USERS")
STEPS: List[int] = (
    [int(x) for x in _steps_env.split(",")] if _steps_env else _default_steps
)
STEP_DURATION = int(os.getenv("STEP_DURATION", 120))


class StepLoadShape(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()
        idx = int(run_time // STEP_DURATION)

        if idx < len(STEPS):
            users = STEPS[idx]
            spawn = max(1, users // 10)
            return users, spawn

        return None


class APIUser(FastHttpUser):
    network_timeout = 180
    connection_timeout = 60
    wait_time = between(0.5, 2)

    def _post(self, url: str, files):
        self.client.post(url, files=files)

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
