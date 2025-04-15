from fastapi.testclient import TestClient
import pytest
import os
from pathlib import Path

# Set the current working directory to the parent directory of the script
current_directory = Path(__file__).parent.parent.parent

# Environment variables
os.environ["USE_GPU"] = "false"
os.environ["MAX_BATCH_SIZE"] = "8"
os.environ["CONFIG_PATH"] = str(current_directory / "resnet" / "onnx" / "configs" / "config.json")

from resnet.onnx.main import app


# Create Test client
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_equal_or_not_equal():
    assert 3 == 3
