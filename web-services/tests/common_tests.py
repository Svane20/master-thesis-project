from fastapi.testclient import TestClient
import pytest
import io
import os
import zipfile
from PIL import Image
from pathlib import Path
from typing import Generator, Any

# Directories
images_directory = Path(__file__).parent / "images"


def run_test_health_endpoint(client: Generator[TestClient, Any, None]):
    """
    This test checks the health endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def run_test_info_endpoint(client: Generator[TestClient, Any, None]):
    """
    This test checks the info endpoint of the FastAPI app.

    Args:
        client (Generator[TestClient, Any, None]): The test client for the FastAPI app.
    """
    response = client.get("/api/v1/info")
    assert response.status_code == 200

    data = response.json()
    expected_deployment = "cuda" if os.environ["USE_GPU"] == "true" else "cpu"

    assert "projectName" in data
    assert "modelType" in data
    assert "deploymentType" in data
    assert data["deploymentType"] == expected_deployment


def run_test_single_inference(client: Generator[TestClient, Any, None]):
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


def run_test_batch_inference(client: Generator[TestClient, Any, None]):
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


def run_test_sky_replacement(client: Generator[TestClient, Any, None]):
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


def run_test_sky_replacement_extra(client: Generator[TestClient, Any, None]):
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


def run_test_batch_sky_replacement(client: Generator[TestClient, Any, None]):
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


def run_test_batch_sky_replacement_extra(client: Generator[TestClient, Any, None]):
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
