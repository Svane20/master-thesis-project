import torch

from pathlib import Path
import platform
import os

from libs.configuration.configuration import load_configuration_and_checkpoint
from libs.datasets.synthetic.data_loaders import create_data_loader
from libs.datasets.synthetic.transforms import get_test_transforms
from libs.evaluation.inference import evaluate_model
from libs.training.utils.logger import setup_logging

from ..build_model import build_model

setup_logging(__name__)


def main() -> None:
    # Directories
    base_directory = Path(__file__).resolve().parent.parent.parent
    if platform.system() == "Windows":
        configuration_path: Path = base_directory / "unet_vgg16/configs/inference_windows.yaml"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = base_directory / "unet_vgg16/configs/inference_linux.yaml"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_configuration_and_checkpoint(configuration_path)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        compile_model=False,
        device=str(device),
        mode="eval"
    )

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


if __name__ == "__main__":
    main()
