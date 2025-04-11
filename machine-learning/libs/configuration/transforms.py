from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TransformsConfig:
    train: Optional[Dict[str, Any]] = field(default_factory=dict)
    val: Optional[Dict[str, Any]] = field(default_factory=dict)
    test: Optional[Dict[str, Any]] = field(default_factory=dict)

    def asdict(self):
        return {k: v for k, v in {"train": self.train, "val": self.val, "test": self.test}.items() if v is not None}
