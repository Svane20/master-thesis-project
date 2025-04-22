import csv
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Tuple, List
import psutil
import signal

from tests.utils.performance import setup_api_instance

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

TESTS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = TESTS_DIR / "reports"
LOCUST_FILE = TESTS_DIR / "utils" / "locust.py"
LOCUST_FILE_DPT = TESTS_DIR / "utils" / "locust_dpt.py"
HOST_URL = "http://localhost:8000"

MODELS: Iterable[str] = ("dpt", "resnet", "swin")
FORMATS: Iterable[str] = ("pytorch", "onnx", "torchscript")
WORKER_COUNTS: Iterable[int] = (1, 2, 4)


def monitor_system(outfile: Path, stop_event: threading.Event, interval: int = 1) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", buffering=1) as f:  # line‑buffered
        w = csv.writer(f)
        w.writerow(["ts", "cpu_%", "ram_MiB", "gpu_id", "gpu_util_%", "gpu_mem_MiB"])
        while not stop_event.is_set():
            ts = time.time()
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().used // 2 ** 20
            gpus = _query_gpu() or [(None, None, None)]
            for gid, util, mem in gpus:
                w.writerow([ts, cpu, ram, gid, util, mem])
            f.flush()
            time.sleep(interval)


def _query_gpu() -> List[Tuple[int, float, float]]:
    """
    Return [(gpu_id, util%, mem_MiB), …].
    Returns an empty list if nvidia‑smi is unavailable or errors.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        return [
            (int(gid), float(util), float(mem))
            for gid, util, mem in (line.split(",") for line in out.strip().splitlines())
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        if not getattr(_query_gpu, "_warned", False):
            print("WARN: nvidia-smi not found – GPU metrics will be empty")
            _query_gpu._warned = True
        return []


def run_locust(model: str, csv_prefix: str, stop_evt: threading.Event) -> None:
    locust_file = LOCUST_FILE_DPT if model == "dpt" else LOCUST_FILE
    csv_path = REPORTS_DIR / csv_prefix

    cmd = [
        "locust", "-f", str(locust_file),
        "--headless", "--host", HOST_URL,
        "--csv-full-history", f"--csv={csv_path}",
    ]
    print("Running:\n  " + " \\\n  ".join(cmd))

    locust_proc = subprocess.Popen(cmd, text=True)

    # ---- watch memory every second ----------------------------------
    while locust_proc.poll() is None and not stop_evt.is_set():
        if psutil.virtual_memory().percent > 90:
            print("RAM > 90% – stopping this run before OOM")
            locust_proc.send_signal(signal.SIGINT)
            break
        time.sleep(1)

    code = locust_proc.wait()
    if code == 0:
        print("Locust finished OK")
    elif code == (128 + signal.SIGKILL):
        print("WARN: Locust was killed by SIGKILL (likely OOM). "
              "Skipping remaining steps for this combination.")
    else:
        print(f"WARN: Locust exited with code {code}")


def main() -> None:
    for model in MODELS:
        for fmt in FORMATS:
            for hw, use_gpu in ("cpu", False), ("gpu", True):
                for workers in WORKER_COUNTS:
                    tag = f"{model}_{fmt}_{hw}_{workers}"
                    print(f"\n=== {tag.upper()} ===")

                    # Launch API instance under test
                    server, api_thr = setup_api_instance(model, fmt, use_gpu, workers)

                    # Start system‑monitor thread
                    stop_evt = threading.Event()
                    mon_thr = threading.Thread(
                        target=monitor_system,
                        args=(REPORTS_DIR / f"{tag}_sys.csv", stop_evt),
                        daemon=True,
                    )
                    mon_thr.start()

                    # Fire up Locust
                    try:
                        run_locust(model, tag, stop_evt)
                    finally:
                        stop_evt.set()
                        mon_thr.join()

                        # ask uvicorn to shut down
                        server.should_exit = True
                        pause = 120 if model == "dpt" else 30
                        api_thr.join(timeout=pause)

                        if api_thr.is_alive():
                            print("Server still alive – forcing exit")
                            server.force_exit = True  # uvicorn will break its loop
                            api_thr.join(timeout=10)

                        if api_thr.is_alive():
                            print("ERROR: uvicorn thread did not terminate; continuing anyway")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
