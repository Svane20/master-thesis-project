from pydantic.dataclasses import dataclass


@dataclass
class EvaluationConfig:
    checkpoint_path: str
    compile_model: bool
