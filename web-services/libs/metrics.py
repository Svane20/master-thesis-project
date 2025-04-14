from prometheus_client import Gauge, Histogram

MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time to load the model (seconds)")
INFERENCE_TIME = Histogram("inference_duration_seconds", "Time for model inference (seconds)")
SKY_REPLACE_TIME = Histogram("sky_replacement_duration_seconds", "Time for sky replacement compositing (seconds)")
TOTAL_TIME = Histogram("request_total_duration_seconds", "Total request processing time (seconds)")
