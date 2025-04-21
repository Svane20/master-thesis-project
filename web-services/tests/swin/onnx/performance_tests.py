import pytest
import asyncio
from pathlib import Path

from tests.utils.configuration import get_custom_config_path
from tests.utils.testing import get_performance_client

# ————————————————————————————————————————————————————————————
# Configuration
# ————————————————————————————————————————————————————————————
current_directory = Path(__file__).parent
project_name = current_directory.parent.name
model_type = current_directory.name

# Pre‑load one image to avoid reopening in every task
IMAGE_DIR = current_directory.parent.parent / "images"
IMAGE_PATH = IMAGE_DIR / "0001.jpg"
IMAGE_BYTES = IMAGE_PATH.read_bytes()


# ————————————————————————————————————————————————————————————
# Fixtures
# ————————————————————————————————————————————————————————————
@pytest.fixture(scope="function")
def custom_config_path(tmp_path_factory):
    return get_custom_config_path(tmp_path_factory, project_name, model_type)


@pytest.fixture
async def client(request, custom_config_path, monkeypatch):
    async for c in get_performance_client(
            request,
            custom_config_path,
            monkeypatch,
            project_name,
            model_type,
    ):
        yield c


# ————————————————————————————————————————————————————————————
# Test
# ————————————————————————————————————————————————————————————
@pytest.mark.parametrize("client", [False, True], indirect=True)
@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """
    Fire 10 concurrent POST /swin/onnx/api/v1/predict calls,
    each sending the same small JPEG, and expect HTTP 200.
    """

    async def hit_predict():
        files = {"file": ("0001.jpg", IMAGE_BYTES, "image/jpeg")}
        r = await client.post("api/v1/predict", files=files)
        assert r.status_code == 200, f"got {r.status_code}: {await r.aread()}"

    # spawn 10 “virtual users” at once
    tasks = [asyncio.create_task(hit_predict()) for _ in range(10)]
    await asyncio.gather(*tasks)
