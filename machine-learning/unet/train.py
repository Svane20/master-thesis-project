from pathlib import Path

from libs.configuration.configuration import ConfigurationMode
from libs.training.train import start_training

from build_model import build_model_for_train
from utils import load_config


def main() -> None:
    # Define logs directory
    logs_directory = Path(__file__).resolve().parent / "logs"

    # Get the configuration and load model
    config = load_config(ConfigurationMode.Training)
    model = build_model_for_train(config.model)

    # Start training
    start_training(model=model, config=config, logs_directory=logs_directory)


if __name__ == "__main__":
    main()
