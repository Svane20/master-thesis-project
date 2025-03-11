from torch.utils.data import DataLoader

from .synthetic_dataset import SyntheticDataset
from ..transforms import Transform


def create_data_loader(
        root_directory: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        transforms: Transform = None,
        shuffle: bool = True,
        drop_last: bool = True,
) -> DataLoader[SyntheticDataset]:
    dataset = SyntheticDataset(
        root_directory=root_directory,
        transforms=transforms
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, num_workers),
        persistent_workers=True,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
