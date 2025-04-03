from torch.utils.data import DataLoader
import torchvision.transforms as T

import os

from .synthetic_dataset import SyntheticDataset, DatasetPhase
from ...configuration.dataset import DatasetConfig


def create_train_data_loader(
        config: DatasetConfig,
        transforms: T.Compose,
        use_trimap: bool = False,
        use_composition: bool = False,
) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the training phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        transforms (transforms.Compose): Transforms to apply to the data.
        use_trimap (bool): Whether to use trimap images or not.
        use_composition (bool): Whether to use composition images or not.

    Returns:
        DataLoader[SyntheticDataset]: The data loader for the training phase.
    """
    return create_data_loader(
        config=config,
        transforms=transforms,
        phase=DatasetPhase.Train,
        num_workers=config.train.num_workers,
        shuffle=config.train.shuffle,
        drop_last=config.train.drop_last,
        use_trimap=use_trimap,
        use_composition=use_composition,
    )


def create_val_data_loader(config: DatasetConfig, transforms: T.Compose) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the validation phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        transforms (transforms.Compose): Transforms to apply to the data.

    Returns:
        DataLoader[SyntheticDataset]: The data loader for the validation phase.
    """
    return create_data_loader(
        config=config,
        phase=DatasetPhase.Val,
        transforms=transforms,
        num_workers=config.val.num_workers,
        shuffle=config.val.shuffle,
        drop_last=config.val.drop_last
    )


def create_test_data_loader(config: DatasetConfig, transforms: T.Compose) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the testing phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        transforms (transforms.Compose): Transforms to apply to the data.

    Returns:
        DataLoader[SyntheticDataset]: The data loader for the testing phase.
    """
    return create_data_loader(
        config=config,
        phase=DatasetPhase.Test,
        transforms=transforms,
        num_workers=config.test.num_workers,
        shuffle=config.test.shuffle,
        drop_last=config.test.drop_last
    )


def create_data_loader(
        config: DatasetConfig,
        transforms: T.Compose,
        phase: DatasetPhase,
        num_workers: int,
        shuffle: bool = True,
        drop_last: bool = True,
        use_trimap: bool = False,
        use_composition: bool = False
) -> DataLoader[SyntheticDataset]:
    """
    Create a data loader for the given phase.
    
    Args:
        config (DatasetConfig): The configuration of the dataset.
        transforms (transforms.Compose): Transforms to apply to the data.
        phase (str): The phase of the dataset.
        num_workers (int): The number of workers for the data loader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size. Defaults to True.
        use_trimap (bool, optional): Whether to use trimap images. Defaults to False.
        use_composition (bool, optional): Whether to use composition images. Defaults to False.
        
    Returns:
        DataLoader[SyntheticDataset]: The data loader for the given phase.
    """
    root_directory = os.path.join(config.root, config.name)

    dataset = SyntheticDataset(
        root_directory=root_directory,
        transforms=transforms,
        phase=phase,
        use_trimap=use_trimap,
        use_composition=use_composition,
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
