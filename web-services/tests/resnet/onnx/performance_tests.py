import pytest
import os
from pathlib import Path
import json
import time
import csv

from libs.fastapi.settings import reset_settings_cache
from tests.utils.configuration import get_mock_configuration, get_custom_config_path
from tests.utils.testing import get_test_client

# Directories
root_directory = Path(__file__).parent.parent.parent.parent
current_directory = Path(__file__).parent
metrics_directory = Path(__file__).parent.parent.parent / "metrics"

# Project and Model Type
project_name = current_directory.parent.name
model_type = current_directory.name

# Environment Variables
os.environ["MAX_BATCH_SIZE"] = "8"

# CSV file to store performance metrics
MODEL_LOAD_TIMES_CSV = metrics_directory / "model_load_times.csv"


@pytest.fixture(scope="function")
def custom_config_path(tmp_path_factory):
    return get_custom_config_path(tmp_path_factory, project_name, model_type)


@pytest.fixture
def client(request, custom_config_path, monkeypatch):
    yield from get_test_client(request, custom_config_path, monkeypatch, project_name, model_type)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_model_load_performance(use_gpu, tmp_path_factory, monkeypatch):
    """
    Measures average time to load the model (create_app()) over multiple iterations,
    for both CPU and GPU.
    """
    monkeypatch.setenv("USE_GPU", "true" if use_gpu else "false")
    tmp_dir = tmp_path_factory.mktemp("configs")
    config_file = tmp_dir / "config.json"
    mock_config = get_mock_configuration(project_name, model_type)
    config_file.write_text(json.dumps(mock_config))
    monkeypatch.setenv("CONFIG_PATH", str(config_file))

    # Reset settings to pick up new environment each iteration
    reset_settings_cache()
    from resnet.onnx.main import create_app

    iterations = 10
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        app = create_app()  # triggers load_model() in lifespan
        total = time.perf_counter() - start
        times.append(total)
        print(f"[{'GPU' if use_gpu else 'CPU'}] Iter {i + 1}/{iterations}: Load time = {total:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"[{'GPU' if use_gpu else 'CPU'}] Average load time = {avg_time:.4f}s across {iterations} runs")

    file_exists = os.path.exists(MODEL_LOAD_TIMES_CSV)
    with open(MODEL_LOAD_TIMES_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # If CSV doesn't exist, write header
        if not file_exists:
            writer.writerow(["model_name", "model_type", "hardware", "iteration", "load_time_sec", "avg_time_sec"])

        for i, load_time in enumerate(times, start=1):
            writer.writerow([
                project_name,
                model_type,
                "GPU" if use_gpu else "CPU",
                i,
                f"{load_time:.4f}",
                f"{avg_time:.4f}"
            ])

    print(f"Metrics appended to {MODEL_LOAD_TIMES_CSV}")
