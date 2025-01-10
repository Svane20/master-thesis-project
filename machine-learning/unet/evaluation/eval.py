import torch

from pathlib import Path

from datasets.carvana.data_loaders import create_data_loader
from datasets.transforms import get_test_transforms

from evaluation.inference import evaluate_model
from evaluation.utils.configuration import load_config
from training.utils.logger import setup_logging

from unet.build_model import build_unet_model

setup_logging(__name__)


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_config(
        current_directory=root_directory,
        configuration_path="unet/configuration/inference.yaml"
    )

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_model(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        compile_model=False,
        device=str(device),
        mode="eval"
    )

    # Create data loader
    test_directory = root_directory / configuration.dataset.root / configuration.dataset.name / "test"
    transforms = get_test_transforms(configuration.scratch.resolution)
    data_loader = create_data_loader(
        directory=test_directory,
        transforms=transforms,
        batch_size=configuration.dataset.batch_size,
        pin_memory=configuration.dataset.pin_memory,
        num_workers=configuration.dataset.test.num_workers,
        shuffle=configuration.dataset.test.shuffle,
        drop_last=configuration.dataset.test.drop_last,
    )

    # Model evaluation
    evaluate_model(model=model, data_loader=data_loader, device=device)


if __name__ == "__main__":
    main()
