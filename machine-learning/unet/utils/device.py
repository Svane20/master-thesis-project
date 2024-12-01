import torch


def get_device() -> torch.device:
    """
    Get the available device for training.

    Returns:
        torch.device: Device to run the training on.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_torch_compile_backend() -> str:
    """
    Get the torch compile backend.

    Returns:
        str: Torch compile backend.
    """
    return "aot_eager" if torch.cuda.is_available() else "eager"
