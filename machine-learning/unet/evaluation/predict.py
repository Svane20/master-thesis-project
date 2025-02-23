import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image

from datasets.transforms import get_val_transforms
from evaluation.inference import predict_image
from evaluation.utils.configuration import load_config
from evaluation.visualization import save_prediction
from replacements.foreground_estimation import get_foreground_estimation
from replacements.replacement import replace_background
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

    # Get test transforms
    dataset_path = root_directory / configuration.dataset.root / configuration.dataset.name
    transforms = get_val_transforms(configuration.scratch.resolution)

    # Get an image from the test dataset
    image_title = "2025-02-21_17-12-27"
    image_path = dataset_path / "val" / "images" / f"{image_title}_Image_4.png"
    image = np.array(Image.open(image_path).convert("RGB"))

    # Get the mask path
    mask_path = dataset_path / "val" / "masks" / f"{image_title}_SkyMask_4.png"
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

    # Normalize the alpha mask to [0, 1]
    if predicted_mask.shape[2] == 4:
        # The alpha channel is the 4th channel (index 3).
        alpha_mask = predicted_mask[..., 3].astype(np.float64) / 255.0

    # Remove the alpha channel
    alpha_mask = np.squeeze(predicted_mask, axis=0)

    foreground, _ = get_foreground_estimation(
        image_path,
        alpha_mask=alpha_mask,
        save_dir=predictions_directory,
        save_foreground=True,
    )

    new_sky_path = Path(
        "C:\\Users\\svane\\Desktop\\Thesis\\master-thesis-project\\machine-learning\\unet\\replacements\\skies\\new_sky.webp"
    )

    # Perform sky replacement
    replace_background(
        new_sky_path,
        foreground,
        alpha_mask,
        save_dir=predictions_directory,
        save_image=True,
    )


if __name__ == "__main__":
    main()
