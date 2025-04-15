from fastapi.testclient import TestClient
import pytest
import os
from pathlib import Path
import json
import io
from PIL import Image
import zipfile

from libs.fastapi.settings import reset_settings_cache

# Directories
root_directory = Path(__file__).parent.parent.parent
images_directory = Path(__file__).parent.parent / "images"

# Paths
model_path = root_directory / "resnet" / "onnx" / "models" / "resnet_50_512_v1.onnx"

os.environ["MAX_BATCH_SIZE"] = "8"
os.environ["CONFIG_PATH"] = str(root_directory / "resnet" / "onnx" / "configs" / "config.json")


@pytest.fixture(scope="function")
def custom_config_path(tmp_path_factory):
    """
    Creates a temporary config.json file with custom settings for testing.
    Returns the path to that file as a string.
    """
    tmp_dir = tmp_path_factory.mktemp("configs")
    config_file = tmp_dir / "config.json"
    mock_config = {
        "project_info": {
            "project_name": "resnet",
            "model_type": "onnx"
        },
        "model": {
            "model_path": f"{str(model_path)}",
            "transforms": {
                "image_size": [512, 512],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
    # Write out the mock config
    config_file.write_text(json.dumps(mock_config))
    return str(config_file)


@pytest.fixture
def client(request, custom_config_path, monkeypatch):
    """
    1. Points CONFIG_PATH to our temporary config.
    2. Creates the FastAPI app after config is set.
    3. Yields a TestClient with the lifespan triggered.
    """
    # If we parametrize this fixture with [False, True], Pytest sets request.param accordingly
    use_gpu = request.param if hasattr(request, "param") else False

    # Override the environment variables so the app loads the custom config
    monkeypatch.setenv("USE_GPU", "true" if use_gpu else "false")
    monkeypatch.setenv("CONFIG_PATH", custom_config_path)

    # Reset the settings cache to ensure the new environment variable is picked up
    reset_settings_cache()

    from resnet.onnx.main import create_app
    app = create_app()

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("client", [False], indirect=True)  # CPU only
def test_health_endpoint(client):
    """
    This test checks the health endpoint of the FastAPI app.
    It sends a GET request to the /api/v1/health endpoint and checks if the response
    status code is 200 and the response body contains the expected JSON.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_info_endpoint(client):
    """
    This test will run twice:
      1) Once with USE_GPU=false (CPU)
      2) Once with USE_GPU=true (GPU)

    Because we parametrize the 'client' fixture itself.
    """
    response = client.get("/api/v1/info")
    assert response.status_code == 200

    data = response.json()
    expected_deployment = "cuda" if os.environ["USE_GPU"] == "true" else "cpu"
    assert data["deploymentType"] == expected_deployment


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_single_inference(client):
    test_image_path = images_directory / "0001.jpg"
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"

    # Upload the file
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/single-predict", files=files)

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


@pytest.mark.parametrize("client", [False, True], indirect=True)
def test_batch_inference(client):
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
