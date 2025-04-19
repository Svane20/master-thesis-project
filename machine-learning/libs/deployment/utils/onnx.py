import torch
import onnx
from pathlib import Path
import logging
from typing import Tuple

from ..profiling import measure_onnx_latency, measure_memory_usage


def export_to_onnx(
        model: torch.nn.Module,
        model_name: str,
        directory: Path,
        dummy_input: Tuple[torch.Tensor],
        device: torch.device,
        measure_model: bool = False,
) -> None:
    """
    Export the model to ONNX format.

    Args:
        model (torch.nn.Module): Model to export.
        model_name (str): Name of the model.
        directory (Path): Directory to save the ONNX model to.
        dummy_input (Tuple[torch.Tensor]): Dummy input tensors.
        device (torch.device): Device to run the model on.
        measure_model (bool): Whether to measure memory usage.
    """
    logging.info(f"Exporting ONNX model...")

    # Set model to evaluation mode
    model.eval()

    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    onnx_path = directory / f"{model_name}_{str(device)}.onnx"

    try:
        # Export with the Dynamo exporter
        onnx_prog = torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=True,
        )

        # In‑place graph cleanup: constant‑folding, dead‑node & initializer removal
        onnx_prog.optimize()

        # Persist the _optimized_ graph to disk
        onnx_prog.save(str(onnx_path))

        # Run shape inference on the optimized model
        m = onnx.load(onnx_path)
        m = onnx.shape_inference.infer_shapes(m)
        onnx.save_model(m, onnx_path)

    except Exception as e:
        logging.error(f"ONNX export failed: {e}")
        raise

    logging.info(f"Model exported to ONNX at {onnx_path}")

    # Validate the exported model
    onnx_model = load_onnx_model(onnx_path)
    validate_onnx_model(onnx_model)

    # Measure latency of the exported model
    if measure_model and device.type == "cuda":
        measure_onnx_latency(onnx_path, dummy_input)
        measure_memory_usage(dummy_input)

    logging.info(f"Finished exporting ONNX model.")


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
        raise Exception(f"Failed to check the ONNX model: {e}")
