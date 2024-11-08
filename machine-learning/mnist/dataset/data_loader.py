from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from typing import Any, Tuple, List

from dataset.mnist_dataset import get_dataset


def create_data_loaders(
        batch_size: int,
        transform: transforms.Compose,
        num_workers: int = 2,
        pin_memory: bool = True
) -> Tuple[DataLoader[Any], DataLoader[Any], List[str]]:
    """
    Create data loaders for FashionMNIST training and test datasets.

    Args:
        batch_size (int): Batch size for the data loaders.
        transform (transforms.Compose): Transform to apply to the data.
        num_workers (int): Number of workers to use for data loading. Default is 1.
        pin_memory (bool): Whether to pin memory for faster data loading. Default is True.

    Returns:
        Tuple[DataLoader[Dataset], DataLoader[Dataset], List[str]]: Training data loader, test data loader, and class names
    """
    # Get the training and test datasets
    train_data = get_dataset(train=True, transform=transform)
    test_data = get_dataset(train=False, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Create data loaders
    train_dataloader = create_train_data_loader(train_data, batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = create_test_data_loader(test_data, batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, test_dataloader, class_names


def create_train_data_loader(
        train_data: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True
) -> DataLoader[Dataset]:
    """
    Create a data loader for the training data.

    Args:
        train_data (datasets.FashionMNIST): Training data
        batch_size (int): Batch size for the data loader
        shuffle (bool): Whether to shuffle the data. Default is True.
        num_workers (int): Number of workers to use for data loading. Default is 1.
        pin_memory (bool): Whether to pin memory for faster data loading. Default is True.

    Returns:
        DataLoader[Dataset]: Data loader for the training data
    """
    return DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def create_test_data_loader(
        test_data: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 2,
        pin_memory: bool = True,
) -> DataLoader[Dataset]:
    """
    Create a data loader for the test data.

    Args:
        test_data (datasets.FashionMNIST): Training data
        batch_size (int): Batch size for the data loader
        shuffle (bool): Whether to shuffle the data. Default is False.
        num_workers (int): Number of workers to use for data loading. Default is 1.
        pin_memory (bool): Whether to pin memory for faster data loading. Default is True.

    Returns:
        DataLoader[Dataset]: Data loader for the test data
    """
    return DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
