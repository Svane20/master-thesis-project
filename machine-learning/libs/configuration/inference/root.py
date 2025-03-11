from pydantic.dataclasses import dataclass


@dataclass
class InferenceConfig:
    checkpoint_path: str
