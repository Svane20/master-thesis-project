import torch.nn as nn

from typing import Dict, List


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
