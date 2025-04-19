import torch.nn as nn
import torch.nn.utils.prune as prune

import logging


def apply_structured_pruning(model: nn.Module, amount: float = 0.2):
    """
    Apply structured pruning to the model.

    Args:
        model (nn.Module): Model to be pruned.
        amount (float): Amount of weights to prune. Should be between 0 and 1.

    Returns:
        nn.Module: Pruned model.
    """
    # Set the model to evaluation mode
    model.eval()

    # Prune X % of channels in each Conv2d layer based on L1 norm
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
                prune.remove(module, "weight")
                logging.info(f"Structured pruning applied to {name}.")
            except Exception as e:
                logging.warning(f"Could not prune {name}: {e}")

    return model
