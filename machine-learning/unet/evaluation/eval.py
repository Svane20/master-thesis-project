import torch

from pathlib import Path
import os

from torch.utils.data import DataLoader

from datasets.synthetic.synthetic_dataset import SyntheticDataset
from datasets.transforms import get_val_transforms

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
        configuration_path="unet/configuration/inference_windows.yaml"
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

    # Validation transforms
    val_transforms = get_val_transforms(configuration.scratch.resolution)

    # Load the validation dataset
    dataset_path = root_directory / configuration.dataset.root / configuration.dataset.name
    images_directory = dataset_path / "images"
    masks_directory = dataset_path / "masks"
    all_files = sorted(os.listdir(images_directory))
    split_index = int(len(all_files) * 0.8)
    val_files = all_files[split_index:]

    val_dataset = SyntheticDataset(
        image_directory=images_directory,
        mask_directory=masks_directory,
        transforms=val_transforms,
        file_list=val_files
    )

    data_loader = DataLoader(
        val_dataset,
        batch_size=configuration.dataset.batch_size,
        shuffle=configuration.dataset.test.shuffle,
        num_workers=max(1, configuration.dataset.test.num_workers),
        persistent_workers=True,
        pin_memory=configuration.dataset.pin_memory,
        drop_last=configuration.dataset.test.drop_last
    )

    # Model evaluation
    evaluate_model(model=model, data_loader=data_loader, device=device)


if __name__ == "__main__":
    main()
