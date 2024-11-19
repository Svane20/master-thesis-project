import torch

import warnings


def setup_AMP(benchmark: bool = True) -> None:
    """
    Set up Automatic Mixed Precision (AMP) for faster training.

    Args:
        benchmark (bool): Enable if input sizes are constant. Default is True.
    """
    torch.backends.cudnn.benchmark = benchmark  # Enable if input sizes are constant
    torch.backends.cudnn.deterministic = False  # Set False for better performance

    if torch.cuda.is_available():
        # Automatic Mixed Precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    else:
        warnings.warn("Running on CPU. Skipping CUDA-specific optimizations.")
