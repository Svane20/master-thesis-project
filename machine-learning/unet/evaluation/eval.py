import torch

from pathlib import Path
from PIL import Image

from datasets.carvana.data_loaders import create_data_loader
from datasets.transforms import get_test_transforms

from evaluation.inference import evaluate_model, predict_image
from evaluation.utils.configuration import load_config
from evaluation.visualization import save_predictions, save_prediction, remove_background

from unet.build_model import build_model


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent
    current_directory = Path(__file__).resolve().parent
    outputs_directory = current_directory / "outputs"
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
    save_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
        num_batches=1,
        directory=outputs_directory
    )

    # Single image evaluation
    image_path = test_directory / "images" / "cf89c3220bc4_03.jpg"
    image = Image.open(image_path).convert("RGB")

    predicted_mask = predict_image(image=image, model=model, transform=transforms, device=device)
    save_prediction(image=image, predicted_mask=predicted_mask, directory=predictions_directory)
    remove_background(image=image, predicted_mask=predicted_mask, directory=predictions_directory)


if __name__ == "__main__":
    main()
