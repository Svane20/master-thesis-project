import torch

from pathlib import Path
import onnx
import logging


def export_to_onnx(
        model: torch.nn.Module,
        model_name: str,
        directory: Path,
        dummy_input: torch.Tensor,
) -> None:
    """
    Export the model to ONNX format.

    Args:
        model (torch.nn.Module): Model to export.
        model_name (str): Name of the model.
        directory (Path): Directory to save the ONNX model to.
        dummy_input (torch.Tensor): Dummy input tensor.
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}.onnx"

    # Set model to inference mode
    model.eval()

    # Export model to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logging.info(f"Model exported to ONNX at {save_path}")
    except Exception as e:
        logging.error(f"ONNX export failed: {e}")
        raise

    print(f"Model exported to ONNX at {save_path}")

    # Validate the exported model
    onnx_model = load_onnx_model(save_path)
    validate_onnx_model(onnx_model)


def load_onnx_model(onnx_model_path: Path) -> onnx.ModelProto:
    """
    Load an ONNX model.

    Args:
        onnx_model_path (Path): Path to the ONNX model.

    Returns:
        ModelProto: Loaded ONNX model.
    """
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    try:
        onnx_model = onnx.load(onnx_model_path)
        return onnx_model
    except Exception as e:
        raise Exception(f"Failed to load the ONNX model: {e}")


def validate_onnx_model(onnx_model: onnx.ModelProto) -> None:
    """
    Validate an ONNX model.

    Args:
        onnx_model (ModelProto): ONNX model to validate.
    """
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"Failed to validate the ONNX model: {e}")
