import torch
import torch.nn as nn

from typing import Dict, List
from pathlib import Path
import logging


def check_load_state_dict_errors(
        missing_keys: List,
        unexpected_keys: List,
        strict: bool = True,
):
    """
    Check if there are any errors when loading the state dictionary.

    Args:
        missing_keys (list): List of missing keys.
        unexpected_keys (list): List of unexpected keys.
        strict (bool): Whether to strictly enforce that the keys match. Default is True.
    """
    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
        model: nn.Module,
        state_dict: Dict,
        strict: bool = True,
):
    """
    Load the state dictionary into the model.

    Args:
        model (nn.Module): Model to load the state dictionary into.
        state_dict (dict): State dictionary to load.
        strict (bool): Whether to strictly enforce that the keys match. Default is True.
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
    )

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, is_compiled: bool) -> None:
    """
    Load checkpoint for the model.

    Args:
        model (Module): Model to load the checkpoint.
        checkpoint_path (Path): Path to the checkpoint.
        is_compiled (bool): True if model is compiled.

    Exceptions:
        RuntimeError: If missing or unexpected keys in the checkpoint
    """
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError("Checkpoint not found.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = checkpoint["model"]

    # If model has not been compiled, adjust key names
    if not is_compiled:
        model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError("Missing keys in checkpoint.")

    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError("Unexpected keys in checkpoint.")

    logging.info("Loaded checkpoint successfully")
