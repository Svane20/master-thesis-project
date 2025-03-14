import torch
from pathlib import Path
import os

from ..configuration.configuration import Config
from ..configuration.dataset import DatasetConfig
from ..datasets.synthetic.data_loaders import create_data_loader
from ..datasets.transforms import Transform
from ..training.trainer import Trainer


def start_training(
        model: torch.nn.Module,
        config: Config,
        train_transforms: Transform,
        val_transforms: Transform,
        logs_directory: Path
) -> None:
    # Get the configuration values
    dataset_config: DatasetConfig = config.dataset

    # Set up the data loaders
    root_path = os.path.join(config.dataset.root, config.dataset.name)
    train_data_loader = create_data_loader(
        root_directory=os.path.join(root_path, "train"),
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.train.num_workers,
        pin_memory=config.dataset.pin_memory,
        shuffle=config.dataset.train.shuffle,
        drop_last=config.dataset.train.drop_last,
        transforms=train_transforms
    )
    val_data_loader = create_data_loader(
        root_directory=os.path.join(root_path, "val"),
        batch_size=dataset_config.batch_size,
        num_workers=config.dataset.val.num_workers,
        pin_memory=dataset_config.pin_memory,
        shuffle=config.dataset.val.shuffle,
        drop_last=config.dataset.val.drop_last,
        transforms=val_transforms
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        config=config,
        logs_directory=logs_directory,
    )

    try:
        # Run the training
        trainer.run()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
