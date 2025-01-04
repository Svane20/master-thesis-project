from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class CriterionConfig:
    name: str
    weight_dict: Optional[Dict[str, float]]
