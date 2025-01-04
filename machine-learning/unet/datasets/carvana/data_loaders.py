from torch.utils.data import DataLoader, Dataset

import albumentations as A
from pathlib import Path
from typing import Tuple

from datasets.carvana.carvana_dataset import CarvanaDataset
from datasets.transforms import get_train_transforms, get_test_transforms

from configuration.dataset import DatasetConfig
from configuration.scratch import ScratchConfig


def setup_data_loaders(
        scratch_config: ScratchConfig,
        dataset_config: DatasetConfig
) -> Tuple[DataLoader[Dataset], DataLoader[Dataset]]:
    """
    Set up the data loaders for training.

    Args:
        scratch_config (ScratchConfig): Configuration for scratch.
        dataset_config (DatasetConfig): Configuration for the dataset.

    Returns:
        Tuple[DataLoader[Dataset], DataLoader[Dataset]]: Train and test data loaders.
    """
    # Get train and test transforms
    transforms = get_train_transforms(scratch_config.resolution)
    target_transforms = get_test_transforms(scratch_config.resolution)

    # Construct data filepaths
    current_directory = Path(__file__).resolve().parent.parent.parent
    train_directory = current_directory / dataset_config.root / dataset_config.name / "train"
    test_directory = current_directory / dataset_config.root / dataset_config.name / "test"

    # Create the data loaders
    train_data_loader = create_data_loader(
        directory=train_directory,
        transforms=transforms,
        batch_size=dataset_config.batch_size,
        pin_memory=dataset_config.pin_memory,
        num_workers=dataset_config.train.num_workers,
        shuffle=dataset_config.train.shuffle,
        drop_last=dataset_config.train.drop_last,
    )
    test_data_loader = create_data_loader(
        directory=test_directory,
        transforms=target_transforms,
        batch_size=dataset_config.batch_size,
        pin_memory=dataset_config.pin_memory,
        num_workers=dataset_config.test.num_workers,
        shuffle=dataset_config.test.shuffle,
        drop_last=dataset_config.test.drop_last,
    )

    return train_data_loader, test_data_loader


def create_data_loader(
        directory: Path,
        transforms: A.Compose,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        drop_last: bool,
) -> DataLoader[Dataset]:
    """
    Create a data loader for the training data.

    Args:
        directory (Path): Path to the training data directory.
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of workers to use for data loading.
        shuffle (bool): Whether to shuffle the data.
        pin_memory (bool): Whether to pin memory for faster data loading.
        drop_last (bool): Whether to drop the last incomplete batch.
        transforms (albumentations.Compose): Transforms to apply to the image data.

    Returns:
        DataLoader[Dataset]: Data loader for the training data
    """
    dataset = CarvanaDataset(
        image_directory=directory / "images",
        mask_directory=directory / "masks",
        transforms=transforms,
    )

    workers = max(1, num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=True,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
