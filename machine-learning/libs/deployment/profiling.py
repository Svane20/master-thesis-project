import torch
import onnxruntime as ort
import time
import logging
import os


def measure_latency(model, dummy_input, num_runs=100):
    """
    Measure average inference time (latency) for the given model.
    """
    model.eval()
    timings = []

    # Warm-up (5 runs)
    for _ in range(5):
        _ = model(dummy_input)

    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

    avg_latency = sum(timings) / len(timings)
    logging.info(f"Average inference latency: {avg_latency * 1000:.2f} ms over {num_runs} runs")
    return avg_latency


def measure_onnx_latency(onnx_model_path, dummy_input, num_runs=100):
    """
    Measure average inference time (latency) for the ONNX model.

    Args:
        onnx_model_path (str or Path): Path to the exported ONNX model file.
        dummy_input (torch.Tensor): Dummy input tensor for testing.
        num_runs (int): Number of inference runs to average.

    Returns:
        float: Average latency in seconds.
    """
    timings = []
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

    # Create an ONNX Runtime session using both CUDA and CPU providers.
    session = ort.InferenceSession(str(onnx_model_path), opts, providers=providers)

    logging.info(f"ONNX model providers: {session.get_providers()}")

    # Warm-up runs to stabilize performance
    input_name = session.get_inputs()[0].name
    dummy_np = dummy_input.cpu().numpy()
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_np})

    # Measure latency over a number of runs
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_np})
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    avg_latency = sum(timings) / len(timings)
    logging.info(f"Average ONNX inference latency: {avg_latency * 1000:.2f} ms over {num_runs} runs")
    return avg_latency


def measure_memory_usage(dummy_input):
    """
    If using CUDA, measure peak GPU memory usage during a forward pass.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = dummy_input.new_empty(dummy_input.size())  # dummy operation
        memory_before = torch.cuda.memory_allocated()
        # Forward pass if needed (this example uses a no-op)
        _ = dummy_input
        torch.cuda.synchronize()
        memory_peak = torch.cuda.max_memory_allocated()
        logging.info(f"Peak GPU memory allocated: {memory_peak / (1024 ** 2):.2f} MB")
        return memory_peak
    else:
        logging.info("CUDA is not available – skipping GPU memory measurement.")
        return None
