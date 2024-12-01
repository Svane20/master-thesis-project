import onnx

from pathlib import Path

from constants.directories import EXPORT_DIRECTORY
from constants.outputs import TRAINED_MODEL_CHECKPOINT_NAME


def load_onnx_model(
        model_name: str = TRAINED_MODEL_CHECKPOINT_NAME,
        directory: Path = EXPORT_DIRECTORY,
        postfix: str = 'latest',
):
    model = onnx.load(directory / f"{model_name}_{postfix}.onnx")

    # Check if the model is valid
    onnx.checker.check_model(model)
