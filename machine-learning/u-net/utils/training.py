import torch

import warnings


def setup_AMP() -> None:
    """
    Set up Automatic Mixed Precision (AMP) for faster training.
    """
    torch.backends.cudnn.benchmark = True  # Enable if input sizes are constant
    torch.backends.cudnn.deterministic = False  # Set False for better performance

    if torch.cuda.is_available():
        # Automatic Mixed Precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    else:
        warnings.warn("Running on CPU. Skipping CUDA-specific optimizations.")


def get_warmup_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Get a learning rate scheduler with warmup for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to attach the scheduler to.
        warmup_epochs (int): Number of warmup epochs.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler with warmup.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1 / (epoch + 1) if epoch < warmup_epochs else 1
    )
