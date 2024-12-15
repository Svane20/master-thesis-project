import torch

from pathlib import Path
from typing import Tuple
import onnx


def export_to_onnx(
        model: torch.nn.Module,
        device: torch.device,
        model_name: str,
        directory: Path,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),  # (batch_size, channels, height, width),
) -> None:
    """
    Export the model to ONNX format.

    Args:
        model (torch.nn.Module): Model to export.
        device (torch.device): Device to use for export.
        model_name (str): Name of the model.
        directory (Path): Directory to save the ONNX model to.
        input_shape (Tuple[int, int, int, int]): Input shape for the model. Default is (1, 3, 224, 224).
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}.onnx"

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

    # Validate the exported model
    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("Exported model has been validated.")
    except Exception as e:
        print(f"Failed to validate the exported model: {e}")
