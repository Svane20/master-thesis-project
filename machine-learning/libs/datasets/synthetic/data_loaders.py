from torch.utils.data import DataLoader

import os

from .synthetic_dataset import SyntheticDataset
from ...configuration.dataset import DatasetConfig


def create_train_data_loader(config: DatasetConfig, resolution: int) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the training phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        resolution (int): The resolution of the images.

    Returns:
        DataLoader[SyntheticDataset]: The data loader for the training phase.
    """
    return create_data_loader(
        config=config,
        phase="train",
        resolution=resolution,
        num_workers=config.train.num_workers,
        shuffle=config.train.shuffle,
        drop_last=config.train.drop_last
    )


def create_val_data_loader(config: DatasetConfig, resolution: int) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the validation phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        resolution (int): The resolution of the images.

    Returns:
        DataLoader[SyntheticDataset]: The data loader for the validation phase.
    """
    return create_data_loader(
        config=config,
        phase="val",
        resolution=resolution,
        num_workers=config.val.num_workers,
        shuffle=config.val.shuffle,
        drop_last=config.val.drop_last
    )


def create_test_data_loader(config: DatasetConfig, resolution: int) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the testing phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        resolution (int): The resolution of the images.

    Returns:
        DataLoader[SyntheticDataset]: The data loader for the testing phase.
    """
    return create_data_loader(
        config=config,
        phase="test",
        resolution=resolution,
        num_workers=config.test.num_workers,
        shuffle=config.test.shuffle,
        drop_last=config.test.drop_last
    )


def create_data_loader(
        config: DatasetConfig,
        phase: str,
        resolution: int,
        num_workers: int,
        shuffle: bool = True,
        drop_last: bool = True,
) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the given phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        phase (str): The phase of the dataset.
        resolution (int): The resolution of the images.
        num_workers (int): The number of workers for the data loader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size. Defaults to True.
        
    Returns:
        DataLoader[SyntheticDataset]: The data loader for the given phase.
    """
    root_directory = os.path.join(config.root, config.name)

    dataset = SyntheticDataset(
        root_directory=root_directory,
        resolution=resolution,
        phase=phase
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=max(1, num_workers),
        persistent_workers=True,
        pin_memory=config.pin_memory,
        drop_last=drop_last
    )
