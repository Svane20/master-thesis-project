from pydantic.dataclasses import dataclass

@dataclass
class DeploymentConfig:
    checkpoint_path: str
    destination_directory: str
