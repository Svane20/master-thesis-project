import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any
import yaml

from training.trainer import Trainer
from training.utils.train_utils import set_seeds
from unet.build_unet import build_model_for_train
from unet.configuration import ModelConfig


def _load_configuration(configuration_path: Path):
    """
    Load the configuration from the given path.

    Args:
        configuration_path (pathlib.Path): Path to the configuration.
    """
    if not configuration_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_path}")

    with open(configuration_path, "r") as file:
        data = yaml.safe_load(file)

    return data


def _build_model(configuration: Dict[str, Any]) -> nn.Module:
    """
    Build the model using the provided configuration.

    Args:
        configuration (Dict[str, Any]): Model configuration.

    Returns:
        nn.Module: The constructed model.
    """
    model_config = ModelConfig(**configuration)

    return build_model_for_train(model_config)


def _setup_run(
        scratch_config: Dict[str, Any],
        training_config: Dict[str, Any],
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any]):
    """
    Set up the run for training.

    Args:
        scratch_config (Dict[str, Any]): Scratch configuration.
        training_config (Dict[str, Any]): Training configuration.
        model_config (Dict[str, Any]): Model configuration.
        optimizer_config (Dict[str, Any]): Optimizer configuration.
    """
    set_seeds(training_config["seed"])

    model = _build_model(model_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=optimizer_config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_config=optimizer_config,
        training_config=training_config,
    )
    trainer.run()


def main(args):
    current_dir = Path(__file__).resolve().parent.parent
    configuration_path = current_dir / args.config

    config = _load_configuration(configuration_path)
    scratch_config = config["scratch"]
    training_config = config["training"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]

    _setup_run(scratch_config, training_config, model_config, optimizer_config)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="unet/configuration/training.yaml",
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    main(args)
