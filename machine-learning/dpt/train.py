import torch
from pathlib import Path

from libs.configuration.configuration import ConfigurationMode
from libs.configuration.training.root import TrainConfig
from libs.datasets.synthetic.synthetic_dataset import DatasetPhase
from libs.training.train import start_training
from libs.training.utils.train_utils import set_seeds

from dpt.build_model import build_model_for_train
from dpt.utils.config import load_config
from dpt.utils.transforms import get_transforms


def main() -> None:
    # Define logs directory
    logs_directory = Path(__file__).resolve().parent / "logs"

    # Get the configuration
    config = load_config(ConfigurationMode.Training)

    # Get the configuration values
    training_config: TrainConfig = config.training

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for reproducibility
    set_seeds(training_config.seed)

    # Load the model
    model = build_model_for_train(config.model)

    # Get the transforms
    train_transforms = get_transforms(DatasetPhase.Train, config.scratch.resolution, config.scratch.crop_resolution)
    val_transforms = get_transforms(DatasetPhase.Val, config.scratch.resolution)

    # Start training
    start_training(
        model=model,
        config=config,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        logs_directory=logs_directory,
    )


if __name__ == "__main__":
    main()
