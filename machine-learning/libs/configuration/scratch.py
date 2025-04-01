from pydantic.dataclasses import dataclass
from typing import Optional


@dataclass
class ScratchConfig:
    resolution: int
    crop_resolution: Optional[int]

    def asdict(self):
        return {
            "resolution": self.resolution,
            "crop_resolution": self.crop_resolution if self.crop_resolution is not None else None,
        }
