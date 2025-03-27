import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable, Tuple
import json
from dataclasses import dataclass

CORE_LOSS_KEY = "core_loss"
LOSS_REGISTRY: Dict[str, Tuple[Callable, Tuple[str, ...]]] = {}  # name -> (fn, signature)


def register_loss(name: str, signature: Tuple[str, ...]):
    def decorator(fn):
        LOSS_REGISTRY[name] = (fn, signature)
        return fn

    return decorator


@dataclass
class BaseCriterionConfig:
    losses: List[str]
    weight_dict: Optional[Dict[str, float]] = None
    normalize_weights: bool = True


class CriterionBase(nn.Module):
    def __init__(self, config: BaseCriterionConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.losses = config.losses
        self.normalize_weights = config.normalize_weights
        raw_weights = config.weight_dict or {}

        for name in self.losses:
            if name not in LOSS_REGISTRY:
                raise ValueError(f"Loss function '{name}' is not registered.")
            if name not in raw_weights:
                print(f"[Warning] No weight specified for '{name}', defaulting to 1.0")

        self.weights = self._normalize_weights(raw_weights)
        self._register_loss_buffers(device)

    def _normalize_weights(self, raw: Dict[str, float]) -> Dict[str, float]:
        if not self.normalize_weights:
            return {k: raw.get(k, 1.0) for k in self.losses}
        total = sum(raw.get(k, 1.0) for k in self.losses)
        return {k: raw.get(k, 1.0) / total if total > 0 else 0.0 for k in self.losses}

    def _register_loss_buffers(self, device: torch.device):
        for name in self.losses + [CORE_LOSS_KEY]:
            self.register_buffer(f"{name}_value", torch.tensor(0.0, dtype=torch.float32, device=device))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        all_losses = {}
        total = 0.0

        # Clear existing buffers
        self.reset()

        for loss_name in self.losses:
            fn, signature = LOSS_REGISTRY[loss_name]
            args = [preds, targets] + [kwargs[k] for k in signature if k in kwargs]
            loss_value = fn(self, *args)

            weighted = loss_value * self.weights[loss_name]
            all_losses[loss_name] = weighted

            # Update buffer
            buffer_name = f"{loss_name}_value"
            getattr(self, buffer_name).copy_(weighted.detach())
            total += weighted

        # Update core loss buffer
        core_value = total.detach()
        all_losses[CORE_LOSS_KEY] = total
        getattr(self, f"{CORE_LOSS_KEY}_value").copy_(core_value)

        return all_losses

    def log_dict(self) -> Dict[str, float]:
        return {
            k.replace("_value", ""): float(v.item())
            for k, v in self._buffers.items()
            if k.endswith("_value")
        }

    def reset(self) -> None:
        for k in self._buffers:
            if k.endswith("_value"):
                self._buffers[k].zero_()

    def add_loss(self, name: str, fn: Callable, signature: Tuple[str, ...]):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' already registered.")

        LOSS_REGISTRY[name] = (fn, signature)
        self.losses.append(name)
        self.weights[name] = 1.0
        self.register_buffer(f"{name}_value",
                             torch.tensor(0.0, dtype=torch.float32, device=next(self.parameters()).device))

    def dry_run(self, shape=(2, 1, 256, 256), device="cpu") -> Dict[str, float]:
        pred = torch.rand(shape, device=device)
        target = torch.rand(shape, device=device)
        return self(pred, target)

    def get_config(self) -> Dict[str, Union[List[str], Dict[str, float]]]:
        return {
            "losses": self.losses,
            "weight_dict": self.weights,
            "normalize_weights": self.normalize_weights,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.get_config(), f)

    @classmethod
    def from_json(cls, path: str, device: torch.device):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(BaseCriterionConfig(**config), device=device)

    @classmethod
    def from_config(cls, config: Dict, device: torch.device):
        return cls(BaseCriterionConfig(**config), device=device)

    def __repr__(self) -> str:
        weights = ", ".join(f"{k}={v:.3f}" for k, v in self.weights.items())
        return f"{self.__class__.__name__}(losses={self.losses}, weights={{ {weights} }})"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.losses == other.losses and self.weights == other.weights
