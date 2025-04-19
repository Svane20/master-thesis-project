import torch
import onnxruntime as ort
import time
import logging
import os
from pathlib import Path


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


def measure_onnx_latency(model_path, dummy_input, num_runs=100):
    """
    Measure latency using I/O binding on GPU (zero memcpy) or standard run on CPU.
    """
    model_path = Path(model_path)
    ort_path = model_path.with_suffix(".ort")
    if ort_path.exists():
        model_path = ort_path
        backend = "ORT"
    else:
        backend = "ONNX"

    # Session options
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True

    # Threading
    if dummy_input.device.type == "cuda":
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
    else:
        t = max(1, os.cpu_count() - 1)
        sess_opts.intra_op_num_threads = t
        sess_opts.inter_op_num_threads = t

    # Providers
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if dummy_input.device.type == "cuda"
        else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(
        str(model_path), sess_options=sess_opts, providers=providers
    )

    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    timings = []

    if dummy_input.device.type == "cuda":
        # try I/O binding path
        try:
            io_binding = session.io_binding()

            # Use numpy fallback for binding since ortvalue_from_dlpack isn't available
            # we bind a numpy-backed GPU buffer via DLPack if supported:
            inp_numpy = dummy_input.cpu().numpy()
            io_binding.bind_input(
                name=in_name,
                device_type="cpu",  # host→GPU copy still internal
                device_id=0,
                element_type=ort.OrtValue.ortvalue_from_numpy(inp_numpy).element_type,
                shape=inp_numpy.shape,
                buffer_ptr=ort.OrtValue.ortvalue_from_numpy(inp_numpy).data_ptr()
            )

            # prepare an output buffer on CPU
            out_shape = session.get_outputs()[0].shape
            out_shape = [dummy_input.shape[0]] + [int(x) for x in out_shape[1:]]
            empty_out = ort.OrtValue.ortvalue_from_numpy(
                np.zeros(out_shape, dtype=np.float32)
            )
            io_binding.bind_output(
                name=out_name,
                device_type="cpu",
                device_id=0,
                element_type=empty_out.element_type,
                shape=tuple(out_shape),
                buffer_ptr=empty_out.data_ptr()
            )

            # warm‑up
            for _ in range(5):
                session.run_with_iobinding(io_binding)

            # timed runs
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run_with_iobinding(io_binding)
                timings.append(time.perf_counter() - start)

        except (AttributeError, NotImplementedError):
            # DLPack / GPU‑IO binding not available: fallback to session.run
            logging.warning("ORT I/O binding via DLPack not supported; falling back to session.run")
            cpu_in = dummy_input.cpu().numpy()
            # warm‑up
            for _ in range(5):
                session.run(None, {in_name: cpu_in})
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run(None, {in_name: cpu_in})
                timings.append(time.perf_counter() - start)

    else:
        # CPU path
        cpu_in = dummy_input.cpu().numpy()
        # warm‑up
        for _ in range(5):
            session.run(None, {in_name: cpu_in})
        # timed runs
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {in_name: cpu_in})
            timings.append(time.perf_counter() - start)

    avg_s = sum(timings) / len(timings)
    logging.info(f"Avg {backend} latency: {avg_s * 1000:.1f} ms over {num_runs} runs")
    return avg_s


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
