from dataclasses import dataclass


@dataclass
class DatasetLoaderConfig:
    shuffle: bool
    drop_last: bool


@dataclass
class DatasetConfig:
    name: str
    root: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    train: DatasetLoaderConfig
    test: DatasetLoaderConfig
