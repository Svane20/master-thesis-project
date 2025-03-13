from pydantic.dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class CriterionConfig:
    name: str
    weight_dict: Optional[Dict[str, float]]

    def asdict(self):
        return {
            "name": self.name,
            "weight_dict": self.weight_dict if self.weight_dict else None
        }
