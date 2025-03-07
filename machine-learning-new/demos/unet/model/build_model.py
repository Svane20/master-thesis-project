import torch
import torch.nn as nn

from pathlib import Path

from .modeling.mask_decoder import MaskDecoder
from .modeling.image_encoder import ImageEncoder
from .modeling.unet import UNet


def build_model(checkpoint_path: str, device: str = "cuda", mode: str = "eval") -> nn.Module:
    # Build model and move to device
    model = _build()
    model = model.to(device)

    # Load checkpoint
    _load_checkpoint(model, checkpoint_path, is_compiled=False)

    if mode == "eval":
        model.eval()

    return model


def _load_checkpoint(model: nn.Module, checkpoint_path: str, is_compiled: bool) -> None:
    """
    Load checkpoint for the model.

    Args:
        model (Module): Model to load the checkpoint.
        checkpoint_path (Path): Path to the checkpoint.
        is_compiled (bool): True if model is compiled.

    Exceptions:
        RuntimeError: If missing or unexpected keys in the checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError("Checkpoint not found.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # If model has not been compiled, adjust key names
    if not is_compiled:
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
    if missing_keys:
        print(missing_keys)
        raise RuntimeError("Missing keys in checkpoint.")

    if unexpected_keys:
        print(unexpected_keys)
        raise RuntimeError("Unexpected keys in checkpoint.")

    print("Loaded checkpoint successfully")


def _build() -> nn.Module:
    return UNet(
        image_encoder=ImageEncoder(
            pretrained=True,
            freeze_pretrained=True,
        ),
        mask_decoder=MaskDecoder(
            out_channels=1,
            dropout=0.2
        )
    )
