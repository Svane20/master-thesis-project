import torch

from pathlib import Path
from PIL import Image
import platform
import random
import numpy as np

from configuration.configuration import load_configuration_and_checkpoint
from datasets.transforms import get_val_transforms
from evaluation.inference import predict_image
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

    # Get configuration based on OS
    if platform.system() == "Windows":
        configuration_path: Path = root_directory / "unet/configuration/inference_windows.yaml"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = root_directory / "unet/configuration/inference_linux.yaml"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_configuration_and_checkpoint(configuration_path)

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
    dataset_path = Path(configuration.dataset.root) / configuration.dataset.name
    transforms = get_val_transforms(configuration.scratch.resolution)

    # List all image files in the validation images folder
    images_dir = dataset_path / "val" / "images"
    image_files = list(images_dir.glob("*.png"))

    # Choose a random image file
    chosen_image_path = random.choice(image_files)
    image = np.array(Image.open(chosen_image_path).convert("RGB"))

    # Derive the corresponding mask path by replacing "Image" with "SkyMask" in the filename
    mask_filename = chosen_image_path.stem.replace("Image", "SkyMask") + ".png"
    mask_path = dataset_path / "val" / "masks" / mask_filename
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    mask = mask / 255.0  # Scale mask values to the [0, 1] range

    # Predict the mask
    predicted_mask, metrics = predict_image(image=image, mask=mask, model=model, transform=transforms, device=device)

    # Normalize the alpha mask to [0, 1]
    if predicted_mask.shape[2] == 4:
        # Extract the alpha channel properly.
        predicted_mask = predicted_mask[..., 3].astype(np.float64) / 255.0
    else:
        # If there’s no fourth channel, assume the mask is the output.
        predicted_mask = np.squeeze(predicted_mask, axis=0)

    # Upscale the predicted mask to the original image size
    predicted_mask = np.array(Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))).astype(np.float32) / 255.0

    # Save the prediction
    save_prediction(
        image=image,
        predicted_mask=predicted_mask,
        gt_mask=mask,
        metrics=metrics,
        directory=predictions_directory
    )

    # Get the foreground estimation
    foreground, _ = get_foreground_estimation(
        chosen_image_path,
        alpha_mask=predicted_mask,
        save_dir=predictions_directory,
        save_foreground=True,
    )

    # Replace the background with a new sky
    new_sky_path = root_directory / "replacements" / "skies" / "new_sky.webp"
    replace_background(
        new_sky_path,
        foreground,
        predicted_mask,
        save_dir=predictions_directory,
        save_image=True,
    )


if __name__ == "__main__":
    main()
