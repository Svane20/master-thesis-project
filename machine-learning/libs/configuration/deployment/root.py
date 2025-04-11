from pydantic.dataclasses import dataclass


@dataclass
class DeploymentConfig:
    resolution: int
    checkpoint_path: str
    destination_directory: str
