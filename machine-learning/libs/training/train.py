import torch
import os

from ..configuration.configuration import Config, TrainConfig
from ..configuration.dataset import DatasetConfig
from ..configuration.scratch import ScratchConfig
from ..datasets.synthetic.data_loaders import create_data_loader
from ..datasets.synthetic.transforms import get_train_transforms, get_test_transforms
from ..training.trainer import Trainer
from ..training.utils.train_utils import set_seeds


def start_training(model: torch.nn.Module, config: Config) -> None:
    # Get the configuration values
    scratch_config: ScratchConfig = config.scratch
    dataset_config: DatasetConfig = config.dataset
    training_config: TrainConfig = config.training

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for reproducibility
    set_seeds(training_config.seed)

    # Set up the data loaders
    root_path = os.path.join(config.dataset.root, config.dataset.name)
    train_data_loader = create_data_loader(
        root_directory=os.path.join(root_path, "train"),
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.train.num_workers,
        pin_memory=config.dataset.pin_memory,
        shuffle=config.dataset.train.shuffle,
        drop_last=config.dataset.train.drop_last,
        transforms=get_train_transforms(scratch_config.resolution)
    )
    test_data_loader = create_data_loader(
        root_directory=os.path.join(root_path, "test"),
        batch_size=dataset_config.batch_size,
        num_workers=config.dataset.test.num_workers,
        pin_memory=dataset_config.pin_memory,
        shuffle=config.dataset.test.shuffle,
        drop_last=config.dataset.test.drop_last,
        transforms=get_test_transforms(scratch_config.resolution)
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        config=config,
    )

    try:
        # Run the training
        trainer.run()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
