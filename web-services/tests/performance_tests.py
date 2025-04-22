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


def kill_port_8000_tree():
    for conn in psutil.net_connections(kind="inet"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == 8000:
            pid = conn.pid
            if not pid:
                continue
            try:
                parent = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            # kill children first
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            # then kill parent
            try:
                parent.kill()
            except Exception:
                pass


def _monitor_system(outfile: Path, stop_event: threading.Event, interval: int = 1) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", buffering=1) as f:
        w = csv.writer(f)
        w.writerow(["ts", "cpu_%", "ram_MiB", "gpu_id", "gpu_util_%", "gpu_mem_MiB", "gpu_mem_total_MiB"])
        while not stop_event.is_set():
            ts = time.time()
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().used // 2 ** 20
            gpus = _query_gpu()
            if not gpus:
                w.writerow([ts, cpu, ram, None, None, None, None])
            else:
                for gid, util, used, total in gpus:
                    w.writerow([ts, cpu, ram, gid, util, used, total])
            f.flush()
            time.sleep(interval)


def _query_gpu() -> List[Tuple[int, float, float, float]]:
    """
    Return [(gpu_id, util%, mem_MiB), …].
    Returns an empty list if nvidia‑smi is unavailable or errors.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ], encoding="utf-8"
        )
        gpus = []
        for line in out.strip().splitlines():
            gid, util, used, total = [x.strip() for x in line.split(",")]
            gpus.append((int(gid), float(util), float(used), float(total)))
        return gpus
    except (FileNotFoundError, subprocess.CalledProcessError):
        if not getattr(_query_gpu, "_warned", False):
            print("WARN: nvidia-smi not found – GPU metrics will be empty")
            _query_gpu._warned = True
        return []


def _run_locust(model: str, csv_prefix: str, stop_evt: threading.Event, use_gpu: bool) -> None:
    locust_file = LOCUST_FILE_DPT if model == "dpt" else LOCUST_FILE
    csv_path = REPORTS_DIR / csv_prefix

    cmd = [
        "locust", "-f", str(locust_file),
        "--headless", "--host", HOST_URL,
        "--csv-full-history", f"--csv={csv_path}",
    ]
    print("Running:\n  " + " \\\n  ".join(cmd))

    proc = subprocess.Popen(cmd, text=True)
    try:
        while proc.poll() is None and not stop_evt.is_set():
            if psutil.virtual_memory().percent > 90:
                print("RAM > 90% – stopping this run before OOM")
                proc.send_signal(signal.SIGINT)
                break

            if use_gpu:
                for gid, util, used, total in _query_gpu():
                    mem_ratio = (used / total) if total else 0.0
                    if mem_ratio > 0.9:
                        print("VRAM > 90% – stopping this run before OOM")
                        proc.send_signal(signal.SIGINT)
                        stop_evt.set()
                        break

            time.sleep(1)
    finally:
        code = proc.wait()
        if code == 0:
            print("Locust finished OK")
        elif code == (128 + signal.SIGKILL):
            print("WARN: Locust was killed by SIGKILL (likely OOM). "
                  "Skipping remaining steps for this combination.")
        else:
            print(f"WARN: Locust exited with code {code}")


def main() -> None:
    for workers in WORKER_COUNTS:
        model = "dpt"
        fmt = "pytorch"
        hw = "gpu"
        use_gpu = True

        tag = f"{model}_{fmt}_{hw}_{workers}"
        print(f"\n=== {tag.upper()} ===")

        # Launch API instance under test
        server_proc = setup_api_instance(model, fmt, use_gpu, workers)

        # Start system‑monitor thread
        stop_evt = threading.Event()
        mon_thr = threading.Thread(
            target=_monitor_system,
            args=(REPORTS_DIR / f"{tag}_sys.csv", stop_evt),
            daemon=True,
        )
        mon_thr.start()

        # Fire up Locust
        try:
            _run_locust(model, tag, stop_evt, use_gpu)
        finally:
            stop_evt.set()
            mon_thr.join()

            pause = 180 if model == "dpt" else 60
            print("Shutting down server process…")
            server_proc.terminate()

            try:
                server_proc.wait(timeout=pause)
            except subprocess.TimeoutExpired:
                print("WARN: server did not exit in time, killing…")
                server_proc.kill()
                server_proc.wait()

            print("Killing any remaining uvicorn workers…")
            kill_port_8000_tree()

            ram_threshold = 10 * 1024 ** 3
            print("Waiting for RAM to fall below 10 GiB…")
            start = time.time()
            while True:
                used = psutil.virtual_memory().used
                if used <= ram_threshold:
                    break
                if time.time() - start > 300:
                    print("\n  timeout waiting for RAM — proceeding anyway")
                    break
                print(f"\r   RAM used: {used / 1024 ** 3:5.1f} GiB …", end="", flush=True)
                time.sleep(1)
            print("\n  RAM is now under threshold.")

            if use_gpu:
                gpu_threshold = 1024  # in MiB
                print(" Waiting for GPU memory to fall below 1 GiB…")
                start = time.time()
                while True:
                    total_used = sum(mem for _, _, mem, _ in _query_gpu())
                    if total_used <= gpu_threshold:
                        break
                    if time.time() - start > 300:
                        print("\n  timeout waiting for GPU — proceeding anyway")
                        break
                    print(f"\r   GPU used: {total_used:6.1f} MiB …", end="", flush=True)
                    time.sleep(1)
                print("\n  GPU memory is now under threshold.")

            print("  Cleanup complete, next run starting!")

    # for model in MODELS:
    #     for fmt in FORMATS:
    #         for hw, use_gpu in ("cpu", False), ("gpu", True):
    #             for workers in WORKER_COUNTS:
    #                 if model == "dpt" and use_gpu and workers == 4:
    #                     print(
    #                         f"Skipping {model.upper()} with num_workers={workers} since it cannot be on the {hw.upper()}")
    #                     continue
    #
    #                 tag = f"{model}_{fmt}_{hw}_{workers}"
    #                 print(f"\n=== {tag.upper()} ===")
    #
    #                 # Launch API instance under test
    #                 server_proc = setup_api_instance(model, fmt, use_gpu, workers)
    #
    #                 # Start system‑monitor thread
    #                 stop_evt = threading.Event()
    #                 mon_thr = threading.Thread(
    #                     target=_monitor_system,
    #                     args=(REPORTS_DIR / f"{tag}_sys.csv", stop_evt),
    #                     daemon=True,
    #                 )
    #                 mon_thr.start()
    #
    #                 # Fire up Locust
    #                 try:
    #                     _run_locust(model, tag, stop_evt, use_gpu)
    #                 finally:
    #                     stop_evt.set()
    #                     mon_thr.join()
    #
    #                     pause = 180 if model == "dpt" else 60
    #                     print("Shutting down server process…")
    #                     server_proc.terminate()
    #
    #                     try:
    #                         server_proc.wait(timeout=pause)
    #                     except subprocess.TimeoutExpired:
    #                         print("WARN: server did not exit in time, killing…")
    #                         server_proc.kill()
    #                         server_proc.wait()
    #
    #                     print("Killing any remaining uvicorn workers…")
    #                     kill_port_8000_tree()
    #
    #                     ram_threshold = 10 * 1024 ** 3
    #                     print("Waiting for RAM to fall below 10 GiB…")
    #                     start = time.time()
    #                     while True:
    #                         used = psutil.virtual_memory().used
    #                         if used <= ram_threshold:
    #                             break
    #                         if time.time() - start > 300:
    #                             print("\n  timeout waiting for RAM — proceeding anyway")
    #                             break
    #                         print(f"\r   RAM used: {used / 1024 ** 3:5.1f} GiB …", end="", flush=True)
    #                         time.sleep(1)
    #                     print("\n  RAM is now under threshold.")
    #
    #                     if use_gpu:
    #                         gpu_threshold = 1024  # in MiB
    #                         print(" Waiting for GPU memory to fall below 1 GiB…")
    #                         start = time.time()
    #                         while True:
    #                             total_used = sum(mem for _, _, mem, _ in _query_gpu())
    #                             if total_used <= gpu_threshold:
    #                                 break
    #                             if time.time() - start > 300:
    #                                 print("\n  timeout waiting for GPU — proceeding anyway")
    #                                 break
    #                             print(f"\r   GPU used: {total_used:6.1f} MiB …", end="", flush=True)
    #                             time.sleep(1)
    #                         print("\n  GPU memory is now under threshold.")
    #
    #                     print("  Cleanup complete, next run starting!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
