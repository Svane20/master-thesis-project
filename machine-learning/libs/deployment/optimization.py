import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.utils.prune as prune

import logging


def apply_dynamic_quantization(model: nn.Module, target_device: str = "cpu") -> nn.Module:
    """
    Apply dynamic quantization to the model if the target device is CPU.
    Dynamic quantization is only supported on CPU.

    Args:
        model (nn.Module): Model to be quantized.
        target_device (str): The device for which the model is intended ("cpu" or "cuda").

    Returns:
        nn.Module: Quantized model if target_device is "cpu"; else, returns original model.
    """
    model.eval()
    if target_device.lower() != "cpu":
        logging.warning("Dynamic quantization is CPU-only. Skipping quantization for target device '%s'.", target_device)
        return model

    quantized_model = quant.quantize_dynamic(model=model, qconfig_spec={nn.Linear}, dtype=torch.qint8)
    logging.info("Dynamic quantization applied to the model on CPU.")
    return quantized_model


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
