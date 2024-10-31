from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from typing import Any, Tuple, List


def create_data_loaders(batch_size: int) -> Tuple[DataLoader[Any], DataLoader[Any], List[str]]:
    """
    Create data loaders for FashionMNIST training and test datasets.

    Args:
        batch_size (int): Batch size for the data loaders.

    Returns:
        Tuple[DataLoader[Any], DataLoader[Any], List[str]]: Training data loader, test data loader, and class names
    """

    # Get the training and test datasets
    train_data = _get_train_data()
    test_data = _get_test_data()

    # Get class names
    class_names = train_data.classes

    # Create data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader, class_names


def _get_test_data() -> datasets.FashionMNIST:
    """
    Get FashionMNIST test data.

    Returns:
        datasets.FashionMNIST: Test data.
    """
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )


def _get_train_data() -> datasets.FashionMNIST:
    """
    Get FashionMNIST training data.

    Returns:
        datasets.FashionMNIST: Training data.
    """
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
