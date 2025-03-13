import torch

import os

from ..configuration.configuration import Config

from .utils.inference import evaluate_model
from ..datasets.synthetic.data_loaders import create_data_loader
from ..datasets.synthetic.transforms import get_test_transforms
from ..training.utils.logger import setup_logging

setup_logging(__name__)


def run_evaluation(configuration: Config, model: torch.nn.Module, device: torch.device):
    # Create data loader
    data_loader = create_data_loader(
        root_directory=os.path.join(configuration.dataset.root, configuration.dataset.name, "test"),
        batch_size=configuration.dataset.batch_size,
        pin_memory=configuration.dataset.pin_memory,
        num_workers=configuration.dataset.test.num_workers,
        shuffle=configuration.dataset.test.shuffle,
        drop_last=configuration.dataset.test.drop_last,
        transforms=get_test_transforms(configuration.scratch.resolution)
    )

    # Model evaluation
    evaluate_model(model=model, data_loader=data_loader, device=device)
