import torch
from pathlib import Path

from ..configuration.configuration import Config
from ..datasets.synthetic.data_loaders import create_train_data_loader, create_val_data_loader
from ..datasets.transforms import get_transforms
from ..training.trainer import Trainer


def start_training(model: torch.nn.Module, config: Config, logs_directory: Path) -> None:
    """
    Start the training process.

    Args:
        model (torch.nn.Module): The model to be trained.
        config (Config): The configuration object.
        logs_directory (Path): The directory to save the logs.
    """
    # Get the transforms
    transforms = get_transforms(config=config.transforms, phases=["train", "val"])

    # Set up the data loaders
    train_data_loader = create_train_data_loader(config=config.dataset, transforms=transforms["train"])
    val_data_loader = create_val_data_loader(config=config.dataset, transforms=transforms["val"])

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
