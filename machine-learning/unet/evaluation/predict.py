import numpy as np
import torch

from pathlib import Path
from PIL import Image

from datasets.transforms import get_test_transforms
from evaluation.inference import predict_image
from evaluation.utils.configuration import load_config
from evaluation.visualization import save_prediction
from training.utils.logger import setup_logging
from unet.build_model import build_unet_model

setup_logging(__name__)


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent
    current_directory = Path(__file__).resolve().parent
    predictions_directory = current_directory / "predictions"

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

    # Get test transforms
    test_directory = root_directory / configuration.dataset.root / configuration.dataset.name / "test"
    transforms = get_test_transforms(configuration.scratch.resolution)

    # Get an image from the test dataset
    image_title = "cf89c3220bc4_03"
    image_path = test_directory / "images" / f"{image_title}.jpg"
    image = np.array(Image.open(image_path).convert("RGB"))

    # Get the mask path
    mask_path = test_directory / "masks" / f"{image_title}_mask.gif"
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    mask = mask / 255.0  # Scale to [0, 1] range

    # Predict the mask
    predicted_mask, metrics = predict_image(image=image, mask=mask, model=model, transform=transforms, device=device)
    save_prediction(
        image=image,
        predicted_mask=predicted_mask,
        gt_mask=mask,
        metrics=metrics,
        directory=predictions_directory
    )


if __name__ == "__main__":
    main()
