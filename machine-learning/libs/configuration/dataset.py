from pydantic.dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetLoaderConfig:
    num_workers: int
    shuffle: bool
    drop_last: bool

    def asdict(self):
        return {
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last
        }


@dataclass
class TrainingDatasetLoaderConfig(DatasetLoaderConfig):
    num_workers: int
    shuffle: bool
    drop_last: bool
    use_trimap: bool
    use_composition: bool

    def asdict(self):
        return {
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "use_trimap": self.use_trimap,
            "use_composition": self.use_composition
        }


@dataclass
class DatasetConfig:
    name: str
    root: str
    batch_size: int
    pin_memory: bool
    train: Optional[TrainingDatasetLoaderConfig] = None
    val: Optional[DatasetLoaderConfig] = None
    test: Optional[DatasetLoaderConfig] = None

    def asdict(self):
        return {
            "name": self.name,
            "root": self.root,
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "train": self.train.asdict() if self.train else None,
            "val": self.val.asdict() if self.val else None,
            "test": self.test.asdict() if self.test else None,
        }
