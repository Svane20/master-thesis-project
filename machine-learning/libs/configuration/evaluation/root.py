from pydantic.dataclasses import dataclass


@dataclass
class InferenceConfig:
    use_sliding_window: bool
    tile_size: int
    overlap: int


@dataclass
class EvaluationConfig:
    checkpoint_path: str
    compile_model: bool

    inference: InferenceConfig
