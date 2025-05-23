import torch
from pathlib import Path

from libs.configuration.configuration import ConfigurationMode
from libs.configuration.training.root import TrainConfig
from libs.training.train import start_training
from libs.training.utils.train_utils import set_seeds

from resnet.build_model import build_model_for_train
from resnet.utils.config import load_config


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

    # Start training
    start_training(model=model, config=config, logs_directory=logs_directory)


if __name__ == "__main__":
    main()
