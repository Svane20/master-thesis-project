from torch.utils.data import DataLoader, Dataset

import albumentations as A
from pathlib import Path
from typing import Tuple
import os
import random

from datasets.synthetic.synthetic_dataset import SyntheticDataset
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
    # Get transforms for training and validation
    train_transforms = get_train_transforms(scratch_config.resolution)
    val_transforms = get_test_transforms(scratch_config.resolution)

    # Construct data filepaths
    current_directory = Path(__file__).resolve().parent.parent.parent
    dataset_path = current_directory / dataset_config.root / dataset_config.name
    images_directory = dataset_path / "images"
    masks_directory = dataset_path / "masks"

    # Get the list of all image files and shuffle them for randomness.
    all_files = sorted(os.listdir(images_directory))

    # Compute the split index (80% training, 20% validation)
    split_index = int(len(all_files) * 0.8)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # Create dataset instances for training and validation.
    train_dataset = SyntheticDataset(
        image_directory=images_directory,
        mask_directory=masks_directory,
        transforms=train_transforms,
        file_list=train_files
    )
    val_dataset = SyntheticDataset(
        image_directory=images_directory,
        mask_directory=masks_directory,
        transforms=val_transforms,
        file_list=val_files
    )

    # Create data loaders.
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.train.shuffle,
        num_workers=max(1, dataset_config.train.num_workers),
        persistent_workers=True,
        pin_memory=dataset_config.pin_memory,
        drop_last=dataset_config.train.drop_last
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.test.shuffle,
        num_workers=max(1, dataset_config.test.num_workers),
        persistent_workers=True,
        pin_memory=dataset_config.pin_memory,
        drop_last=dataset_config.test.drop_last
    )

    return train_data_loader, val_data_loader
