from pydantic.dataclasses import dataclass


@dataclass
class ScratchConfig:
    resolution: int
