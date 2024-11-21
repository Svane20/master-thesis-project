import torch

from pathlib import Path
from typing import Tuple

from constants.directories import EXPORT_DIRECTORY
from constants.outputs import TRAINED_MODEL_CHECKPOINT_NAME
from model.unet import UNetV0
from utils.checkpoints import load_model_checkpoint
from utils.device import get_device


def export_to_onnx(
        model: torch.nn.Module,
        device: torch.device,
        model_name: str = TRAINED_MODEL_CHECKPOINT_NAME,
        directory: Path = EXPORT_DIRECTORY,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),  # (batch_size, channels, height, width),
        postfix: str = 'latest',
):
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}_{postfix}.onnx"

    # Set model to inference mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Export model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
    )

    print(f"Model exported to ONNX at {save_path}")


if __name__ == "__main__":
    target_device = get_device()

    model_name = TRAINED_MODEL_CHECKPOINT_NAME

    # Load trained model
    loaded_model = UNetV0(in_channels=3, out_channels=1, dropout=0.5)
    loaded_model, _ = load_model_checkpoint(model=loaded_model, model_name=model_name, device=target_device)

    # Export model to ONNX
    export_to_onnx(loaded_model, model_name=model_name, device=target_device)
