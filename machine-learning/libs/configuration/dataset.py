from pydantic.dataclasses import dataclass


@dataclass
class DatasetLoaderConfig:
    num_workers: int
    shuffle: bool
    drop_last: bool


@dataclass
class DatasetConfig:
    name: str
    root: str
    batch_size: int
    pin_memory: bool
    train: DatasetLoaderConfig
    test: DatasetLoaderConfig
