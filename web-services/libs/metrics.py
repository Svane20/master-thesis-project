from prometheus_client import Histogram

MODEL_LOAD_TIME = Histogram(
    name="model_startup_latency_seconds",
    documentation="Model startup latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
SINGLE_INFERENCE_TIME = Histogram(
    name="single_inference_latency_seconds",
    documentation="Single inference latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
SINGLE_INFERENCE_TOTAL_TIME = Histogram(
    name="single_inference_total_time_seconds",
    documentation="Single inference total time (seconds)",
    labelnames=["model", "type", "hardware"],
)
BATCH_INFERENCE_TIME = Histogram(
    name="batch_inference_latency_seconds",
    documentation="Batch inference latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
BATCH_INFERENCE_TOTAL_TIME = Histogram(
    name="batch_inference_total_time_seconds",
    documentation="Batch inference total time (seconds)",
    labelnames=["model", "type", "hardware"],
)
SKY_REPLACEMENT_INFERENCE_TIME = Histogram(
    name="sky_replacement_inference_latency_seconds",
    documentation="Sky replacement inference latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
SKY_REPLACEMENT_TIME = Histogram(
    name="sky_replacement_latency_seconds",
    documentation="Sky replacement latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
SKY_REPLACEMENT_TOTAL_TIME = Histogram(
    name="sky_replacement_total_time_seconds",
    documentation="Sky replacement total time (seconds)",
    labelnames=["model", "type", "hardware"],
)
BATCH_SKY_REPLACEMENT_INFERENCE_TIME = Histogram(
    name="batch_sky_replacement_inference_latency_seconds",
    documentation="Batch sky replacement inference latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
BATCH_SKY_REPLACEMENT_TIME = Histogram(
    name="batch_sky_replacement_latency_seconds",
    documentation="Batch sky replacement latency (seconds)",
    labelnames=["model", "type", "hardware"],
)
BATCH_SKY_REPLACEMENT_TOTAL_TIME = Histogram(
    name="batch_sky_replacement_total_time_seconds",
    documentation="Batch sky replacement total time (seconds)",
    labelnames=["model", "type", "hardware"],
)
