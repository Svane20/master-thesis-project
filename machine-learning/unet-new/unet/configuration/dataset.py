from dataclasses import dataclass


@dataclass
class DatasetLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    drop_last: bool


@dataclass
class DatasetConfig:
    name: str
    root: str
    train: DatasetLoaderConfig
    test: DatasetLoaderConfig
