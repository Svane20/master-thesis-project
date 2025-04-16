import pytest
from pathlib import Path

from tests.common_tests import run_test_health_endpoint, run_test_info_endpoint, run_test_single_inference, \
    run_test_batch_inference, run_test_sky_replacement, run_test_sky_replacement_extra
from tests.utils.configuration import get_custom_config_path
from tests.utils.testing import get_test_client

# Project and Model Type
current_directory = Path(__file__).parent
project_name = current_directory.parent.name
model_type = current_directory.name


@pytest.fixture(scope="function")
def custom_config_path(tmp_path_factory):
    return get_custom_config_path(tmp_path_factory, project_name, model_type)


@pytest.fixture
def client(request, custom_config_path, monkeypatch):
    yield from get_test_client(request, custom_config_path, monkeypatch, project_name, model_type)


@pytest.mark.parametrize("client", [False], indirect=True)  # CPU only
def test_health_endpoint(client):
    run_test_health_endpoint(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_info_endpoint(client):
    run_test_info_endpoint(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_single_inference(client):
    run_test_single_inference(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_inference(client):
    run_test_batch_inference(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_sky_replacement(client):
    run_test_sky_replacement(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_sky_replacement_extra(client):
    run_test_sky_replacement_extra(client)
