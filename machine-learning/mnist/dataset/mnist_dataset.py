from torchvision import datasets
from torchvision.transforms import transforms

from constants.directories import DATA_DIRECTORY


def get_test_data(transform: transforms.Compose) -> datasets.FashionMNIST:
    """
    Get FashionMNIST test data.

    Args:
        transform (transforms.Compose): Transform to apply to the data.

    Returns:
        datasets.FashionMNIST: Test data.
    """
    return datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=transform
    )


def get_train_data(transform: transforms.Compose) -> datasets.FashionMNIST:
    """
    Get FashionMNIST training data.

    Args:
        transform (transforms.Compose): Transform to apply to the data.

    Returns:
        datasets.FashionMNIST: Training data.
    """
    return datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transform
    )
