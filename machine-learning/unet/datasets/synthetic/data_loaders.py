from torch.utils.data import DataLoader, Dataset
import albumentations as A
from pathlib import Path
from typing import Tuple
from datasets.synthetic.synthetic_dataset import SyntheticDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from configuration.dataset import DatasetConfig
from configuration.scratch import ScratchConfig

def setup_data_loaders(
        scratch_config: ScratchConfig,
        dataset_config: DatasetConfig
) -> Tuple[DataLoader[Dataset], DataLoader[Dataset]]:
    # Get train and validation transforms.
    transforms = get_train_transforms(scratch_config.resolution)
    target_transforms = get_val_transforms(scratch_config.resolution)

    # Construct data filepaths.
    current_directory = Path(__file__).resolve().parent.parent.parent
    train_directory = current_directory / dataset_config.root / dataset_config.name / "train"
    val_directory = current_directory / dataset_config.root / dataset_config.name / "val"

    # Create the data loaders.
    train_data_loader = create_data_loader(
        directory=train_directory,
        transforms=transforms,
        batch_size=dataset_config.batch_size,
        pin_memory=dataset_config.pin_memory,
        num_workers=dataset_config.train.num_workers,
        shuffle=dataset_config.train.shuffle,
        drop_last=dataset_config.train.drop_last,
    )
    val_data_loader = create_data_loader(
        directory=val_directory,
        transforms=target_transforms,
        batch_size=dataset_config.batch_size,
        pin_memory=dataset_config.pin_memory,
        num_workers=dataset_config.val.num_workers,
        shuffle=dataset_config.val.shuffle,
        drop_last=dataset_config.val.drop_last,
    )

    return train_data_loader, val_data_loader

def create_data_loader(
        directory: Path,
        transforms: A.Compose,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        drop_last: bool,
) -> DataLoader[Dataset]:
    dataset = SyntheticDataset(
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
