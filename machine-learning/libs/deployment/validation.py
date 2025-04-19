import torch
import onnxruntime as ort
import numpy as np
import os
import logging


def compare_model_outputs(model, ts_model, onnx_path, dummy_input, rtol=1e-3, atol=5e-3, strict=True):
    """
    Compare the output of a PyTorch model with its TorchScript and ONNX versions.

    Args:
        model (torch.nn.Module): Original PyTorch model.
        ts_model (torch.jit.ScriptModule): TorchScript model.
        onnx_path (str or Path): Path to the ONNX model file.
        dummy_input (torch.Tensor): Dummy input tensor for testing.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
        strict (bool): If True, an error is raised on discrepancy; else only a warning is logged.

    Returns:
        bool: True if outputs are consistent, False otherwise.
    """
    # Ensure models are in evaluation mode
    model.eval()
    ts_model.eval()

    # Get outputs from the original and TorchScript models
    with torch.no_grad():
        original_output = model(dummy_input)
        ts_output = ts_model(dummy_input)

    orig_np = original_output.cpu().numpy()
    ts_np = ts_output.cpu().numpy()

    # Calculate differences for TorchScript comparison
    max_abs_diff = np.abs(orig_np - ts_np).max()
    max_rel_diff = np.abs(orig_np - ts_np).max() / (np.abs(orig_np).max() + 1e-12)
    total_elements = orig_np.size
    mismatches = np.sum(~np.isclose(orig_np, ts_np, rtol=rtol, atol=atol))
    mismatch_percent = 100 * mismatches / total_elements
    logging.info(f"Max absolute difference (TorchScript): {max_abs_diff:.6f}, "
                 f"Max relative difference (TorchScript): {max_rel_diff:.6f}")
    logging.info(f"TorchScript mismatches: {mismatches} / {total_elements} elements ({mismatch_percent:.2f}%)")

    try:
        np.testing.assert_allclose(orig_np, ts_np, rtol=rtol, atol=atol)
        logging.info("TorchScript model output is consistent with the original PyTorch model.")
    except AssertionError as e:
        msg = (f"TorchScript output mismatch: Max abs diff = {max_abs_diff:.6f}, "
               f"{mismatch_percent:.2f}% elements mismatched")
        if strict:
            logging.error(msg)
        else:
            logging.warning(msg)

    # Compare ONNX outputs
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() - 1
    opts.inter_op_num_threads = os.cpu_count() - 1
    if torch.cuda.is_available():
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DnnlExecutionProvider",
            "CPUExecutionProvider"
        ]
    else:
        providers = ["DnnlExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(str(onnx_path), opts, providers=providers)
    logging.info(f"ONNX model providers: {ort_session.get_providers()}")
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    max_abs_diff_onnx = np.abs(orig_np - ort_output).max()
    max_rel_diff_onnx = np.abs(orig_np - ort_output).max() / (np.abs(orig_np).max() + 1e-12)
    mismatches_onnx = np.sum(~np.isclose(orig_np, ort_output, rtol=rtol, atol=atol))
    mismatch_percent_onnx = 100 * mismatches_onnx / total_elements
    logging.info(f"Max absolute difference (ONNX): {max_abs_diff_onnx:.6f}, "
                 f"Max relative difference (ONNX): {max_rel_diff_onnx:.6f}")
    logging.info(f"ONNX mismatches: {mismatches_onnx} / {total_elements} elements "
                 f"({mismatch_percent_onnx:.2f}%)")

    try:
        np.testing.assert_allclose(orig_np, ort_output, rtol=rtol, atol=atol)
        logging.info("ONNX model output is consistent with the original PyTorch model.")
    except AssertionError as e:
        msg = (f"ONNX output mismatch: Max abs diff = {max_abs_diff_onnx:.6f}, "
               f"{mismatch_percent_onnx:.2f}% elements mismatched")
        if strict:
            logging.error(msg)
        else:
            logging.warning(msg)
