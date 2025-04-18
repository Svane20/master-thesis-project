from enum import Enum
from pathlib import Path
import json
from typing import Dict, Any

from _pytest.tmpdir import TempPathFactory


class Project(str, Enum):
    ResNet = "resnet"
    Swin = "swin"
    DPT = "dpt"


class ModelType(str, Enum):
    ONNX = "onnx"
    TorchScript = "torchscript"
    Pytorch = "pytorch"


def get_custom_config_path(tmp_path_factory: TempPathFactory, project_name: str, model_type: str) -> str:
    """
    Creates a temporary config.json file with custom settings for testing.
    Returns the path to that file as a string.

    Args:
        tmp_path_factory (TempPathFactory): Factory for creating temporary directories.
        project_name (Project): An Enum or similar representing the project (resnet, dpt, etc.).
        model_type (ModelType): An Enum or similar representing the model type (onnx, torchscript, pytorch).

    Returns:
        str: The path to the temporary config.json file.
    """
    tmp_dir = tmp_path_factory.mktemp("configs")
    config_file = tmp_dir / "config.json"
    mock_config = get_mock_configuration(project_name, model_type)
    config_file.write_text(json.dumps(mock_config))
    return str(config_file)


def get_mock_configuration(project_name: str, model_type: str) -> Dict[str, Any]:
    """
    Loads a configuration from the 'config.json' located under:
        <root> / <project_name> / <model_type> / configs / config.json

    Then extracts the 'model_path' from the JSON, and rewrites it to point to
    the actual file path under:
        <root> / <project_name> / <model_type> / models / <model_filename>

    Raises:
        FileNotFoundError: If either the config.json or the model file is missing.
        ValueError:        If config.json is not valid JSON.

    Args:
        project_name (Project): An Enum or similar representing the project (resnet, dpt, etc.).
        model_type (ModelType): An Enum or similar representing the model type (onnx, torchscript, pytorch).

    Returns:
        Dict[str, Any]: The loaded configuration dictionary with an updated, absolute model path.
    """
    # 1) Determine root directory and config path
    root_directory = Path(__file__).resolve().parent.parent.parent
    config_path = root_directory / project_name / model_type / "configs" / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # 2) Parse the JSON data
    try:
        config_json = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file '{config_path}': {e}")

    # 3) Extract the filename from model_path in the config
    relative_model_path = config_json["model"]["model_path"]
    model_filename = Path(relative_model_path).name  # e.g. "resnet_50_512_v1.onnx"

    # 4) Build the absolute path to the model
    absolute_model_path = (
            root_directory
            / project_name
            / model_type
            / "models"
            / model_filename
    )

    # 5) Update the config to point to the correct absolute model path
    config_json["model"]["model_path"] = str(absolute_model_path)

    # 6) Return the final config
    return config_json
