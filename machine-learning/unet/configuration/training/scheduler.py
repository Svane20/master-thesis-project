from pydantic.dataclasses import dataclass
from typing import Dict, Union


@dataclass
class SchedulerConfig:
    name: str
    enabled: bool
    parameters: Dict[str, Union[int, float, str]]

    def asdict(self):
        return {
            "name": self.name,
            "enabled": self.enabled,
            "parameters": self.parameters
        }
