from fastapi.testclient import TestClient
import torch
import os
import time
import csv
import statistics
from pathlib import Path
import logging

from libs.fastapi.settings import reset_settings_cache
from tests.utils.configuration import get_custom_config_path
from tests.utils.factory import get_create_app_func

# Set up logging (ensure these messages appear in your console)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Directories
images_directory = Path(__file__).parent / "images"
metrics_directory = Path(__file__).parent / "metrics"
metrics_directory.mkdir(exist_ok=True)  # ensure directory exists

# CSV file to store performance metrics
MODEL_LOAD_TIMES_CSV = metrics_directory / "model_load_times.csv"
SINGLE_INFERENCE_TIMES_CSV = metrics_directory / "single_inference_times.csv"
BATCH_INFERENCE_TIMES_CSV = metrics_directory / "batch_inference_times.csv"
SKY_REPLACEMENT_TIMES_CSV = metrics_directory / "sky_replacement_times.csv"

# Constants
MODEL_WARMUP_ITERATIONS = 1
MODEL_LOAD_ITERATIONS = 10

WARMUP_EPOCHS = 5
EPOCHS = 100


def run_model_load_performance_test(use_gpu, tmp_path_factory, monkeypatch, project_name, model_type):
    """
    Measures average time to load the model (via create_app()) over multiple iterations,
    for both CPU and GPU. Also logs additional statistics.

    Parameters:
        use_gpu (bool): Whether to test on GPU or CPU.
        tmp_path_factory: Factory to generate temporary paths.
        monkeypatch: Fixture to modify environment variables.
        project_name (str): Name of the model project (e.g., 'swin', 'resnet', 'dpt').
        model_type (str): Format type ('onnx', 'torchscript', 'pytorch').
    """
    # Prepare configuration file path
    config_file = get_custom_config_path(tmp_path_factory, project_name, model_type)

    # Set environment variables for configuration
    monkeypatch.setenv("USE_GPU", "true" if use_gpu else "false")
    monkeypatch.setenv("CONFIG_PATH", config_file)
    monkeypatch.setenv("MAX_BATCH_SIZE", "8")

    # Reset settings to pick up new environment values
    reset_settings_cache()

    # Perform warm-up iterations
    for _ in range(MODEL_WARMUP_ITERATIONS):
        _ = get_create_app_func(project_name, model_type)()
        if use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure GPU tasks are finished

    times = []
    for i in range(MODEL_LOAD_ITERATIONS):
        start = time.perf_counter()

        with TestClient(get_create_app_func(project_name, model_type)()):
            if use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure GPU tasks have completed

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"[{'GPU' if use_gpu else 'CPU'}] Iter {i + 1}/{MODEL_LOAD_ITERATIONS}: Load time = {elapsed:.4f}s")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = statistics.median(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"Performance test completed.")
    print(f"Average time: {avg_time:.4f}s | Min: {min_time:.4f}s | Max: {max_time:.4f}s")
    print(f"Median: {median_time:.4f}s | Standard Deviation: {stdev_time:.4f}s")

    # Append results to CSV file
    file_exists = os.path.exists(MODEL_LOAD_TIMES_CSV)
    with open(MODEL_LOAD_TIMES_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # If CSV doesn't exist, write header
        if not file_exists:
            writer.writerow([
                "model_name", "model_type", "hardware", "iteration", "load_time_sec",
                "avg_time_sec", "min_time_sec", "max_time_sec", "median_time_sec", "stddev_time_sec"
            ])
        for i, load_time in enumerate(times, start=1):
            writer.writerow([
                project_name,
                model_type,
                "GPU" if use_gpu else "CPU",
                i,
                f"{load_time:.4f}",
                f"{avg_time:.4f}",
                f"{min_time:.4f}",
                f"{max_time:.4f}",
                f"{median_time:.4f}",
                f"{stdev_time:.4f}"
            ])

    print(f"Metrics appended to {MODEL_LOAD_TIMES_CSV}")


def run_test_single_inference_performance(client: TestClient, use_gpu, project_name, model_type):
    """
    Test the performance of single inference for the model.

    Args:
        client (TestClient): The test client for the FastAPI app.
        use_gpu (bool): Whether to test on GPU or CPU.
        project_name (str): Name of the model project (e.g., 'swin', 'resnet', 'dpt').
        model_type (str): Format type ('onnx', 'torchscript', 'pytorch').
    """
    test_image_path = images_directory / "0001.jpg"
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"

    # Warmup phase
    print(f"Starting {WARMUP_EPOCHS} warmup iterations...")
    for _ in range(WARMUP_EPOCHS):
        with open(test_image_path, "rb") as f:
            files = {"file": (test_image_path.name, f, "image/jpeg")}
            response = client.post("/api/v1/single-predict", files=files)

    # Performance measurement phase
    times = []
    print(f"Starting {EPOCHS} measured iterations...")
    for i in range(EPOCHS):
        start = time.perf_counter()
        with open(test_image_path, "rb") as f:
            files = {"file": (test_image_path.name, f, "image/jpeg")}
            response = client.post("/api/v1/single-predict", files=files)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Iteration {i + 1}/{EPOCHS}: Inference time = {elapsed:.4f}s")

    # Compute statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = statistics.median(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"Performance test completed.")
    print(f"Average time: {avg_time:.4f}s | Min: {min_time:.4f}s | Max: {max_time:.4f}s")
    print(f"Median: {median_time:.4f}s | Standard Deviation: {stdev_time:.4f}s")

    # Append results to CSV file
    file_exists = SINGLE_INFERENCE_TIMES_CSV.exists()
    with open(SINGLE_INFERENCE_TIMES_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write a header row if the file did not already exist
        if not file_exists:
            writer.writerow(["model_name", "model_type", "hardware", "iteration", "inference_time_sec", "avg_time_sec",
                             "min_time_sec", "max_time_sec", "median_time_sec", "stddev_time_sec"])
        for i, t in enumerate(times, start=1):
            writer.writerow([project_name, model_type, "GPU" if use_gpu else "CPU", i, f"{t:.4f}", f"{avg_time:.4f}",
                             f"{min_time:.4f}", f"{max_time:.4f}", f"{median_time:.4f}", f"{stdev_time:.4f}"])

    print(f"Metrics appended to {SINGLE_INFERENCE_TIMES_CSV}")


def run_test_batch_inference_performance(client: TestClient, use_gpu, project_name, model_type):
    """
    Test the performance of batch inference for the model.

    Args:
        client (TestClient): The test client for the FastAPI app.
        use_gpu (bool): Whether to test on GPU or CPU.
        project_name (str): Name of the model project (e.g., 'swin', 'resnet', 'dpt').
        model_type (str): Format type ('onnx', 'torchscript', 'pytorch').
    """
    image_paths = [
        images_directory / "0001.jpg",
        images_directory / "0055.jpg",
        images_directory / "0086.jpg",
        images_directory / "0211.jpg",
        images_directory / "1901.jpg",
        images_directory / "2022.jpg",
        images_directory / "2041.jpg",
        images_directory / "10406.jpg",
    ]
    for img_path in image_paths:
        assert img_path.exists(), f"Test image not found: {img_path}"

    # Build the request
    files_data = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            content = f.read()
        files_data.append(("files", (img_path.name, content, "image/jpeg")))

    # Warmup phase
    print(f"Starting {WARMUP_EPOCHS} warmup iterations...")
    for _ in range(WARMUP_EPOCHS):
        response = client.post("/api/v1/batch-predict", files=files_data)

    # Performance measurement phase
    times = []
    print(f"Starting {EPOCHS} measured iterations...")
    for i in range(EPOCHS):
        start = time.perf_counter()
        response = client.post("/api/v1/batch-predict", files=files_data)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Iteration {i + 1}/{EPOCHS}: Inference time = {elapsed:.4f}s")

    # Compute statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = statistics.median(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"Performance test completed.")
    print(f"Average time: {avg_time:.4f}s | Min: {min_time:.4f}s | Max: {max_time:.4f}s")
    print(f"Median: {median_time:.4f}s | Standard Deviation: {stdev_time:.4f}s")

    # Append results to CSV file
    file_exists = BATCH_INFERENCE_TIMES_CSV.exists()
    with open(BATCH_INFERENCE_TIMES_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write a header row if the file did not already exist
        if not file_exists:
            writer.writerow(["model_name", "model_type", "hardware", "iteration", "inference_time_sec", "avg_time_sec",
                             "min_time_sec", "max_time_sec", "median_time_sec", "stddev_time_sec"])
        for i, t in enumerate(times, start=1):
            writer.writerow([project_name, model_type, "GPU" if use_gpu else "CPU", i, f"{t:.4f}", f"{avg_time:.4f}",
                             f"{min_time:.4f}", f"{max_time:.4f}", f"{median_time:.4f}", f"{stdev_time:.4f}"])

    print(f"Metrics appended to {BATCH_INFERENCE_TIMES_CSV}")


def run_test_sky_replacement_performance(client: TestClient, use_gpu, project_name, model_type):
    """
    Test the performance of sky replacement for the model.

    Args:
        client (TestClient): The test client for the FastAPI app.
        use_gpu (bool): Whether to test on GPU or CPU.
        project_name (str): Name of the model project (e.g., 'swin', 'resnet', 'dpt').
        model_type (str): Format type ('onnx', 'torchscript', 'pytorch').
    """
    test_image_path = images_directory / "0001.jpg"
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"

    # Warmup phase
    print(f"Starting {WARMUP_EPOCHS} warmup iterations...")
    for _ in range(WARMUP_EPOCHS):
        with open(test_image_path, "rb") as f:
            files = {"file": (test_image_path.name, f, "image/jpeg")}
            response = client.post("/api/v1/sky-replacement", files=files)

    # Performance measurement phase
    times = []
    print(f"Starting {EPOCHS} measured iterations...")
    for i in range(EPOCHS):
        start = time.perf_counter()
        with open(test_image_path, "rb") as f:
            files = {"file": (test_image_path.name, f, "image/jpeg")}
            response = client.post("/api/v1/sky-replacement", files=files)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Iteration {i + 1}/{EPOCHS}: Inference time = {elapsed:.4f}s")

    # Compute statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = statistics.median(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"Performance test completed.")
    print(f"Average time: {avg_time:.4f}s | Min: {min_time:.4f}s | Max: {max_time:.4f}s")
    print(f"Median: {median_time:.4f}s | Standard Deviation: {stdev_time:.4f}s")

    # Append results to CSV file
    file_exists = SKY_REPLACEMENT_TIMES_CSV.exists()
    with open(SKY_REPLACEMENT_TIMES_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write a header row if the file did not already exist
        if not file_exists:
            writer.writerow(["model_name", "model_type", "hardware", "iteration", "inference_time_sec", "avg_time_sec",
                             "min_time_sec", "max_time_sec", "median_time_sec", "stddev_time_sec"])
        for i, t in enumerate(times, start=1):
            writer.writerow([project_name, model_type, "GPU" if use_gpu else "CPU", i, f"{t:.4f}", f"{avg_time:.4f}",
                             f"{min_time:.4f}", f"{max_time:.4f}", f"{median_time:.4f}", f"{stdev_time:.4f}"])

    print(f"Metrics appended to {SKY_REPLACEMENT_TIMES_CSV}")
