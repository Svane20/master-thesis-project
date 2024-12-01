from torch.utils.data import DataLoader, Dataset

import albumentations as A
from typing import Tuple
from pathlib import Path
import os

from dataset.carvana_dataset import CarvanaDataset

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def create_data_loaders(
        train_directory: Path,
        test_directory: Path,
        batch_size: int,
        transform: A.Compose,
        target_transform: A.Compose,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True
) -> Tuple[DataLoader[Dataset], DataLoader[Dataset]]:
    """
    Create data loaders for training and test datasets.

    Args:
        train_directory (Path): Path to the training data directory.
        test_directory (Path): Path to the test data directory.
        batch_size (int): Batch size for the data loaders.
        transform (albumentations.Compose): Transform to apply to the image data.
        target_transform (albumentations.Compose): Transform to apply to the mask data.
        num_workers (int): Number of workers to use for data loading. Default is 1.
        pin_memory (bool): Whether to pin memory for faster data loading. Default is True.

    Returns:
        Tuple[DataLoader[Dataset], DataLoader[Dataset]]: Training data loader, test data loader
    """
    train_dataloader = create_train_data_loader(
        train_directory,
        batch_size,
        transform=transform,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataloader = create_test_data_loader(
        test_directory,
        batch_size,
        transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader


def create_train_data_loader(
        directory: Path,
        batch_size: int,
        transform: A.Compose,
        shuffle: bool = True,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True
) -> DataLoader[Dataset]:
    """
    Create a data loader for the training data.

    Args:
        directory (Path): Path to the training data directory.
        batch_size (int): Batch size for the data loader
        transform (albumentations.Compose): Transform to apply to the image data.
        shuffle (bool): Whether to shuffle the data. Default is True.
        num_workers (int): Number of workers to use for data loading. Default is 1.
        pin_memory (bool): Whether to pin memory for faster data loading. Default is True.

    Returns:
        DataLoader[Dataset]: Data loader for the training data
    """
    dataset = CarvanaDataset(
        image_directory=directory / "images",
        mask_directory=directory / "masks",
        transform=transform,
    )

    workers = max(1, num_workers)
    persistent_workers = workers > 0 and pin_memory

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )


def create_test_data_loader(
        directory: Path,
        batch_size: int,
        transform: A.Compose,
        shuffle: bool = False,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True
) -> DataLoader[Dataset]:
    """
    Create a data loader for the test data.

    Args:
        directory (Path): Path to the training data directory.
        batch_size (int): Batch size for the data loader
        transform (albumentations.Compose): Transform to apply to the image data.
        batch_size (int): Batch size for the data loader
        shuffle (bool): Whether to shuffle the data. Default is False.
        num_workers (int): Number of workers to use for data loading. Default is 1.
        pin_memory (bool): Whether to pin memory for faster data loading. Default is True.

    Returns:
        DataLoader[Dataset]: Data loader for the test data
    """
    dataset = CarvanaDataset(
        image_directory=directory / "images",
        mask_directory=directory / "masks",
        transform=transform,
    )

    workers = max(1, num_workers)
    persistent_workers = workers > 0 and pin_memory

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )
