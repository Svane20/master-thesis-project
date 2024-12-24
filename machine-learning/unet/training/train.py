import torch
import torch.optim as optim

from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets.carvana.data_loaders import setup_data_loaders

from training.criterions import MattingLoss
from training.optimizer import construct_optimizer
from training.trainer import Trainer
from training.utils.train_utils import set_seeds

from unet.build_model import build_model_for_train
from unet.configuration.configuration import ModelConfig, Config, load_configuration
from unet.configuration.dataset import DatasetConfig
from unet.configuration.scratch import ScratchConfig
from unet.configuration.training.base import TrainConfig


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
    model_config: ModelConfig = config.model

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for reproducibility
    set_seeds(training_config.seed)

    # Construct model, criterion, optimizer and scheduler
    model = build_model_for_train(model_config)
    optimizer = construct_optimizer(model, training_config.optimizer)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.scheduler.t_max,
        eta_min=training_config.scheduler.eta_min
    )

    # Set up the data loaders
    train_data_loader, test_data_loader = setup_data_loaders(scratch_config, dataset_config)

    # Set up the trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        training_config=training_config,
    )

    try:
        # Run the training
        trainer.run()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")


def main(args: Namespace) -> None:
    """
    Args:
        args (Namespace): Command-line arguments.
    """
    current_dir = Path(__file__).resolve().parent.parent
    configuration_path = current_dir / args.config

    config: Config = load_configuration(configuration_path)

    _setup_run(config)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="unet/configs/training.yaml",
        help="Path to the configuration file.",
    )

    arguments = parser.parse_args()
    main(arguments)
