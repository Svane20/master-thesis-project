import torch
from pathlib import Path

from libs.configuration.configuration import ConfigurationMode
from libs.configuration.training.root import TrainConfig
from libs.training.train import start_training
from libs.training.utils.train_utils import set_seeds

from build_model import build_model_for_train
from utils import load_config
from dataset.transforms import get_train_transforms, get_val_transforms


def main() -> None:
    # Define logs directory
    logs_directory = Path(__file__).resolve().parent / "logs"

    # Get the configuration and load model
    config = load_config(ConfigurationMode.Training)
    model = build_model_for_train(config.model)

    # Get the configuration values
    training_config: TrainConfig = config.training

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for reproducibility
    set_seeds(training_config.seed)

    # Get the transforms
    train_transforms = get_train_transforms(resolution=config.scratch.resolution)
    val_transforms = get_val_transforms(resolution=config.scratch.resolution)

    # Start training
    start_training(
        model=model,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        config=config,
        logs_directory=logs_directory
    )


if __name__ == "__main__":
    main()
