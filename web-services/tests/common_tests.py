import pytest
from fastapi.testclient import TestClient
import logging
from typing import Iterable, Generator, Any
from pathlib import Path
import io
import zipfile
from PIL import Image

from tests.utils.configuration import get_custom_config_path
from tests.utils.testing import get_test_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODELS: Iterable[str] = ("dpt", "resnet", "swin")
FORMATS: Iterable[str] = ("onnx", "pytorch", "torchscript")

# Directories
images_directory = Path(__file__).parent / "images"


@pytest.fixture(params=MODELS)
def model(request):
    return request.param


@pytest.fixture(params=FORMATS)
def fmt(request):
    return request.param


@pytest.fixture(scope="function")
def custom_config_path(tmp_path_factory, model, fmt):
    return get_custom_config_path(tmp_path_factory, model, fmt)


@pytest.fixture
def client(request, custom_config_path, monkeypatch, model, fmt):
    yield from get_test_client(request, custom_config_path, monkeypatch, model, fmt)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_health_endpoint(client):
    _run_test_health_endpoint(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_live_endpoint(client):
    _run_test_live_endpoint(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_info_endpoint(client, model, fmt):
    _run_test_info_endpoint(client, client.use_gpu, model, fmt)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_single_inference(client):
    _run_test_single_inference(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_inference(client):
    _run_test_batch_inference(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_sky_replacement(client):
    _run_test_sky_replacement(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_sky_replacement_extra(client):
    _run_test_sky_replacement_extra(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_sky_replacement(client):
    _run_test_batch_sky_replacement(client)


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_sky_replacement_extra(client):
    _run_test_batch_sky_replacement_extra(client)


def _run_test_health_endpoint(client: Generator[TestClient, Any, None]):
    """
    This test checks the health endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _run_test_live_endpoint(client: Generator[TestClient, Any, None]):
    """
    This test checks the live endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    response = client.get("/api/v1/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def _run_test_info_endpoint(client: Generator[TestClient, Any, None], use_gpu: bool, model: str, fmt: str):
    """
    This test checks the info endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    response = client.get("/api/v1/info")
    assert response.status_code == 200

    data = response.json()

    assert "projectName" in data
    assert "modelType" in data
    assert "deploymentType" in data
    assert data["projectName"] == model
    assert data["modelType"] == fmt
    assert data["deploymentType"] == "cuda" if use_gpu else "cpu"


def _run_test_single_inference(client: Generator[TestClient, Any, None]):
    """
    This test checks the single inference endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    test_image_path = images_directory / "0001.jpg"
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"

    # Upload the file
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

    # Check the response
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert response.headers.get("Content-Type") == "image/png", (
        f"Expected PNG response, got {response.headers.get('Content-Type')}"
    )

    # Validate the returned image is a valid PNG
    returned_bytes = response.content
    with io.BytesIO(returned_bytes) as img_data:
        try:
            result_img = Image.open(img_data)
            result_img.verify()

            img_data.seek(0)
            with Image.open(img_data) as reopened:
                print(f"Returned a {reopened.mode} image of size {reopened.size}")

        except Exception as e:
            pytest.fail(f"Returned file is not a valid image: {e}")


def _run_test_batch_inference(client: Generator[TestClient, Any, None]):
    """
    This test checks the batch inference endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
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
    files_list = []
    for img_path in image_paths:
        f = open(img_path, "rb")
        files_list.append(("files", (img_path.name, f, "image/jpeg")))

    response = client.post("/api/v1/batch-predict", files=files_list)

    # Check the response
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert response.headers.get("Content-Type") == "application/zip", (
        f"Expected ZIP response, got {response.headers.get('Content-Type')}"
    )

    # Parse the returned ZIP file
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes, "r") as zipped:
        namelist = zipped.namelist()
        assert len(namelist) == len(image_paths), (
            f"Expected {len(image_paths)} items in the zip, but got {len(namelist)}"
        )

        # For each file in the ZIP, verify it is a valid PNG
        for name in namelist:
            with zipped.open(name) as png_file:
                try:
                    img = Image.open(png_file)
                    img.verify()

                    png_file.seek(0)
                    with Image.open(png_file) as reopened:
                        print(f"Extracted {name}: {reopened.mode} image, size {reopened.size}")
                except Exception as e:
                    pytest.fail(f"File {name} in the ZIP is not a valid PNG: {e}")


def _run_test_sky_replacement(client: Generator[TestClient, Any, None]):
    """
    This test checks the sky replacement endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    test_image_path = images_directory / "0001.jpg"
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"

    # Upload the file
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/sky-replacement", files=files)

    # Check the response
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert response.headers.get("Content-Type") == "image/png", (
        f"Expected PNG response, got {response.headers.get('Content-Type')}"
    )

    # Validate the returned image is a valid PNG
    returned_bytes = response.content
    with io.BytesIO(returned_bytes) as img_data:
        try:
            result_img = Image.open(img_data)
            result_img.verify()

            img_data.seek(0)
            with Image.open(img_data) as reopened:
                print(f"Returned a {reopened.mode} image of size {reopened.size}")

        except Exception as e:
            pytest.fail(f"Returned file is not a valid image: {e}")


def _run_test_sky_replacement_extra(client: Generator[TestClient, Any, None]):
    """
    This test checks the sky replacement extra endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    test_image_path = images_directory / "0001.jpg"
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"

    # Upload the file
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/sky-replacement?extra=true", files=files)

    # Check the response
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert response.headers.get("Content-Type") == "application/zip", (
        f"Expected ZIP response, got {response.headers.get('Content-Type')}"
    )

    expected_files_len = 3

    # Parse the returned ZIP file
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes, "r") as zipped:
        namelist = zipped.namelist()
        assert len(namelist) == expected_files_len, (
            f"Expected {expected_files_len} items in the zip, but got {len(namelist)}"
        )

        # For each file in the ZIP, verify it is a valid PNG
        for name in namelist:
            with zipped.open(name) as png_file:
                try:
                    img = Image.open(png_file)
                    img.verify()

                    png_file.seek(0)
                    with Image.open(png_file) as reopened:
                        print(f"Extracted {name}: {reopened.mode} image, size {reopened.size}")
                except Exception as e:
                    pytest.fail(f"File {name} in the ZIP is not a valid PNG: {e}")


def _run_test_batch_sky_replacement(client: Generator[TestClient, Any, None]):
    """
    This test checks the batch sky replacement endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
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
    files_list = []
    for img_path in image_paths:
        f = open(img_path, "rb")
        files_list.append(("files", (img_path.name, f, "image/jpeg")))

    response = client.post("/api/v1/batch-sky-replacement", files=files_list)

    # Check the response
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert response.headers.get("Content-Type") == "application/zip", (
        f"Expected ZIP response, got {response.headers.get('Content-Type')}"
    )

    # Parse the returned ZIP file
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes, "r") as zipped:
        namelist = zipped.namelist()
        assert len(namelist) == len(image_paths), (
            f"Expected {len(image_paths)} items in the zip, but got {len(namelist)}"
        )

        # For each file in the ZIP, verify it is a valid PNG
        for name in namelist:
            with zipped.open(name) as png_file:
                try:
                    img = Image.open(png_file)
                    img.verify()

                    png_file.seek(0)
                    with Image.open(png_file) as reopened:
                        print(f"Extracted {name}: {reopened.mode} image, size {reopened.size}")
                except Exception as e:
                    pytest.fail(f"File {name} in the ZIP is not a valid PNG: {e}")


def _run_test_batch_sky_replacement_extra(client: Generator[TestClient, Any, None]):
    """
    This test checks the batch sky replacement extra endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
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
    files_list = []
    for img_path in image_paths:
        f = open(img_path, "rb")
        files_list.append(("files", (img_path.name, f, "image/jpeg")))

    response = client.post("/api/v1/batch-sky-replacement?extra=true", files=files_list)

    # Check the response
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert response.headers.get("Content-Type") == "application/zip", (
        f"Expected ZIP response, got {response.headers.get('Content-Type')}"
    )

    expected_files_len = len(image_paths) * 3  # 3 files per image

    # Parse the returned ZIP file
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes, "r") as zipped:
        namelist = zipped.namelist()
        assert len(namelist) == expected_files_len, (
            f"Expected {expected_files_len} items in the zip, but got {len(namelist)}"
        )

        # For each file in the ZIP, verify it is a valid PNG
        for name in namelist:
            with zipped.open(name) as png_file:
                try:
                    img = Image.open(png_file)
                    img.verify()

                    png_file.seek(0)
                    with Image.open(png_file) as reopened:
                        print(f"Extracted {name}: {reopened.mode} image, size {reopened.size}")
                except Exception as e:
                    pytest.fail(f"File {name} in the ZIP is not a valid PNG: {e}")
