import torch

from .utils.inference import evaluate_model
from ..configuration.configuration import Config
from ..datasets.synthetic.data_loaders import create_test_data_loader
from ..training.utils.logger import setup_logging

setup_logging(__name__)


def run_evaluation(configuration: Config, model: torch.nn.Module, device: torch.device):
    # Create data loader
    data_loader = create_test_data_loader(config=configuration.dataset, resolution=configuration.scratch.resolution)

    # Model evaluation
    evaluate_model(model=model, data_loader=data_loader, device=device)
