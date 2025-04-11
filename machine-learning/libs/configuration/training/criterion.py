from pydantic.dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class CriterionConfig:
    losses: List[str]
    normalize_weights: bool
    weight_dict: Optional[Dict[str, float]] = None

    def asdict(self):
        return {
            "losses": self.losses,
            "normalize_weights": self.normalize_weights,
            "weight_dict": self.weight_dict if self.weight_dict else None,
        }
