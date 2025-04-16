import pytest
from pathlib import Path

from tests.benchmark_tests import run_model_load_performance_test, run_test_single_inference_performance, \
    run_test_sky_replacement_performance, run_test_batch_inference_performance, \
    run_test_batch_sky_replacement_performance
from tests.utils.configuration import get_custom_config_path
from tests.utils.testing import get_test_client

# Directories
current_directory = Path(__file__).parent

# Project and Model Type
project_name = current_directory.parent.name
model_type = current_directory.name


@pytest.fixture(scope="function")
def custom_config_path(tmp_path_factory):
    return get_custom_config_path(tmp_path_factory, project_name, model_type)


@pytest.fixture
def client(request, custom_config_path, monkeypatch):
    yield from get_test_client(request, custom_config_path, monkeypatch, project_name, model_type)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_model_load_performance(use_gpu, tmp_path_factory, monkeypatch):
    run_model_load_performance_test(use_gpu, tmp_path_factory, monkeypatch, project_name, model_type)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_single_inference_performance(client):
    run_test_single_inference_performance(client, client.use_gpu, project_name, model_type)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_inference_performance(client):
    run_test_batch_inference_performance(client, client.use_gpu, project_name, model_type)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_sky_replacement_performance(client):
    run_test_sky_replacement_performance(client, client.use_gpu, project_name, model_type)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_sky_replacement_performance(client):
    run_test_batch_sky_replacement_performance(client, client.use_gpu, project_name, model_type)
