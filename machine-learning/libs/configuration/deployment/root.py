from pydantic.dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DeploymentConfig:
    resolution: int
    hardware_acceleration: str
    checkpoint_path: str
    destination_directory: str
    optimizations: Optional[Dict[str, Any]] = None

    def asdict(self):
        return {
            "resolution": self.resolution,
            "hardware_acceleration": self.hardware_acceleration,
            "checkpoint_path": self.checkpoint_path,
            "destination_directory": self.destination_directory,
            "optimizations": self.optimizations if self.optimizations is not None else None,
        }
