from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from typing import Any, Tuple, List

from dataset.mnist_dataset import get_train_data, get_test_data


def create_data_loaders(batch_size: int, transform: transforms.Compose) -> Tuple[
    DataLoader[Any], DataLoader[Any], List[str]]:
    """
    Create data loaders for FashionMNIST training and test datasets.

    Args:
        batch_size (int): Batch size for the data loaders.
        transform (transforms.Compose): Transform to apply to the data.

    Returns:
        Tuple[DataLoader[Any], DataLoader[Any], List[str]]: Training data loader, test data loader, and class names
    """
    # Get the training and test datasets
    train_data = get_train_data(transform)
    test_data = get_test_data(transform)

    # Get class names
    class_names = train_data.classes

    # Create data loaders
    train_dataloader = create_train_data_loader(train_data, batch_size)
    test_dataloader = create_test_data_loader(test_data, batch_size)

    return train_dataloader, test_dataloader, class_names


def create_train_data_loader(train_data: datasets.FashionMNIST, batch_size: int) -> DataLoader[Any]:
    """
    Create a data loader for the training data.

    Args:
        train_data (datasets.FashionMNIST): Training data
        batch_size (int): Batch size for the data loader

    Returns:
        DataLoader[Any]: Data loader for the training data
    """
    return DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )


def create_test_data_loader(test_data: datasets.FashionMNIST, batch_size: int) -> DataLoader[Any]:
    """
    Create a data loader for the training data.

    Args:
        test_data (datasets.FashionMNIST): Training data
        batch_size (int): Batch size for the data loader

    Returns:
        DataLoader[Any]: Data loader for the training data
    """
    return DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
