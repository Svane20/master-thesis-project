from pydantic.dataclasses import dataclass


@dataclass
class ScratchConfig:
    resolution: int

    def asdict(self):
        return {
            "resolution": self.resolution
        }
