import torch
import torchvision.transforms as T
from pathlib import Path

from ..configuration.configuration import Config
from ..datasets.synthetic.data_loaders import create_train_data_loader, create_val_data_loader
from ..training.trainer import Trainer


def start_training(
        model: torch.nn.Module,
        config: Config,
        train_transforms: T.Compose,
        val_transforms: T.Compose,
        logs_directory: Path,
        use_trimap: bool = False,
        use_composition: bool = False,
) -> None:
    """
    Start the training process.

    Args:
        model (torch.nn.Module): The model to be trained.
        config (Config): The configuration object.
        train_transforms (transforms.Compose): The transforms to apply to the training data.
        val_transforms (transforms.Compose): The transforms to apply to the validation data.
        logs_directory (Path): The directory to save the logs.
        use_trimap (bool): Whether to use trimap images or not.
        use_composition (bool): Whether to use composition images or not.
    """
    # Set up the data loaders
    train_data_loader = create_train_data_loader(
        config=config.dataset,
        transforms=train_transforms,
        use_trimap=use_trimap,
        use_composition=use_composition,
    )
    val_data_loader = create_val_data_loader(config=config.dataset, transforms=val_transforms)

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
