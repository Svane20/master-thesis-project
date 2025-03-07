import torch

from pathlib import Path
import platform

from datasets.synthetic.data_loaders import setup_data_loaders

from libs.training.trainer import Trainer
from libs.training.utils.train_utils import set_seeds

from configuration.model import ModelConfig
from libs.configuration.configuration import Config, load_configuration
from libs.configuration.dataset import DatasetConfig
from libs.configuration.scratch import ScratchConfig
from libs.configuration.training.root import TrainConfig
from unet.build_model import build_unet_model_for_train


def _setup_run(config: Config) -> None:
    """
    Set up the run for training.

    Args:
        config (Config): Configuration for the training.
    """
    # Get the configuration values
    scratch_config: ScratchConfig = config.scratch
    dataset_config: DatasetConfig = config.dataset
    training_config: TrainConfig = config.training

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for reproducibility
    set_seeds(training_config.seed)

    # Construct model
    model = build_unet_model_for_train(config.model)

    # Set up the data loaders
    train_data_loader, val_data_loader = setup_data_loaders(scratch_config, dataset_config)

    # Set up the trainer
    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        training_config=training_config,
    )

    try:
        # Run the training
        trainer.run()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")


def main() -> None:
    base_directory = Path(__file__).resolve().parent.parent
    if platform.system() == "Windows":
        configuration_path: Path = base_directory / "unet/configs/training_windows.yaml"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = base_directory / "unet/configs/training_linux.yaml"

    config: Config = load_configuration(configuration_path)

    _setup_run(config)


if __name__ == "__main__":
    main()
