from torchvision import datasets
from torchvision.transforms import ToTensor


def get_test_data(root: str = "data") -> datasets.FashionMNIST:
    """
    Get FashionMNIST test data.

    Args:
        root (str): Root directory to save the data.

    Returns:
        datasets.FashionMNIST: Test data.
    """
    return datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor()
    )


def get_train_data(root: str = "data") -> datasets.FashionMNIST:
    """
    Get FashionMNIST training data.

    Args:
        root (str): Root directory to save the data.

    Returns:
        datasets.FashionMNIST: Training data.
    """
    return datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor()
    )
