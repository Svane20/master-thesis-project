import torch


def get_device() -> torch.device:
    """
    Get the available device for training.

    Returns:
        torch.device: Device to run the training on.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
