from torchvision import datasets
from torchvision.transforms import ToTensor

from constants.directories import DATA_DIRECTORY


def get_test_data() -> datasets.FashionMNIST:
    """
    Get FashionMNIST test data.

    Returns:
        datasets.FashionMNIST: Test data.
    """
    return datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=ToTensor()
    )


def get_train_data() -> datasets.FashionMNIST:
    """
    Get FashionMNIST training data.

    Returns:
        datasets.FashionMNIST: Training data.
    """
    return datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=ToTensor()
    )
