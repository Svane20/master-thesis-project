import csv
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Tuple
import psutil

from tests.utils.performance import setup_api_instance

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

TESTS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = TESTS_DIR / "reports"
LOCUST_FILE = TESTS_DIR / "utils" / "locust.py"
HOST_URL = "http://localhost:8000"

MODELS: Iterable[str] = ("resnet", "swin")
FORMATS: Iterable[str] = ("pytorch", "onnx", "torchscript")
WORKER_COUNTS: Iterable[int] = (1, 2, 4)
RUN_TIME = "10m"

def monitor_system(outfile: Path, stop_event: threading.Event, interval: int = 1) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "cpu_%", "ram_MiB", "gpu_id", "gpu_util_%", "gpu_mem_MiB"])
        while not stop_event.is_set():
            ts = time.time()
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().used // 2 ** 20
            gpus = list(_query_gpu()) or [(None, None, None)]
            for gid, gutil, gmem in gpus:
                w.writerow([ts, cpu, ram, gid, gutil, gmem])
            time.sleep(interval)


def _query_gpu() -> Iterable[Tuple[int, float, float]]:
    """Yield (gpu_id, util%, mem_MiB) for each visible device.
    Returns empty list if nvidia‑smi not available."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    for line in out.strip().splitlines():
        gpu_id, util, mem = (x.strip() for x in line.split(","))
        yield int(gpu_id), float(util), float(mem)


def run_locust(run_time: str, csv_prefix: str) -> None:
    csv_path = REPORTS_DIR / f"{csv_prefix}"
    cmd = [
        "locust",
        "-f",
        str(LOCUST_FILE),
        "--headless",
        "--run-time",
        run_time,
        "--host",
        HOST_URL,
        f"--csv={csv_path}",
    ]
    print("Running:", " \
    ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    for model in MODELS:
        for fmt in FORMATS:
            for hw, use_gpu in ("cpu", False), ("gpu", True):
                for workers in WORKER_COUNTS:
                    tag = f"{model}_{fmt}_{hw}_{workers}"
                    print(f"\n=== {tag.upper()} ===")

                    # Launch API instance under test
                    server = setup_api_instance(model, fmt, use_gpu, workers)

                    # Start system‑monitor thread
                    stop_evt = threading.Event()
                    mon_thr = threading.Thread(
                        target=monitor_system,
                        args=(REPORTS_DIR / f"{tag}_sys.csv", stop_evt),
                        daemon=True,
                    )
                    mon_thr.start()

                    # Fire up Locust (step‑load test defined in locustfile.py)
                    try:
                        run_locust(RUN_TIME, csv_prefix=tag)
                    finally:
                        # Shutdown monitor + API regardless of Locust outcome
                        stop_evt.set()
                        mon_thr.join()
                        server.should_exit = True
                        time.sleep(10)  # graceful server close


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
