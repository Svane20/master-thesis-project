import torch

from pathlib import Path
from PIL import Image

from datasets.transforms import get_test_transforms
from evaluation.inference import predict_image
from evaluation.utils.configuration import load_config
from evaluation.visualization import save_prediction, remove_background
from training.utils.logger import setup_logging
from unet.build_model import build_model

setup_logging(__name__)


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent
    current_directory = Path(__file__).resolve().parent
    predictions_directory = current_directory / "predictions"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_config(
        current_directory=root_directory,
        configuration_path="unet/configs/inference.yaml"
    )

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
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
    image_path = test_directory / "images" / "cf89c3220bc4_03.jpg"
    image = Image.open(image_path).convert("RGB")

    # Predict the mask
    predicted_mask = predict_image(image=image, model=model, transform=transforms, device=device)
    save_prediction(image=image, predicted_mask=predicted_mask, directory=predictions_directory)
    remove_background(image=image, predicted_mask=predicted_mask, directory=predictions_directory)


if __name__ == "__main__":
    main()
