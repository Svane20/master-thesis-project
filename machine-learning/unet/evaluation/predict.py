from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from libs.configuration.configuration import ConfigurationMode
from libs.evaluation.predict import run_prediction
from libs.utils.device import get_device

from unet.dataset.transforms import get_test_transforms
from unet.build_model import build_model_for_evaluation
from unet.utils import load_config_and_checkpoint_path


def main() -> None:
    # Directories
    current_directory = Path(__file__).resolve().parent
    predictions_directory = current_directory / "predictions"
    predictions_directory.mkdir(parents=True, exist_ok=True)

    configuration, checkpoint_path = load_config_and_checkpoint_path(ConfigurationMode.Evaluation)

    # Load the model
    device = get_device()
    model = build_model_for_evaluation(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Create transforms
    transforms = get_test_transforms(configuration.scratch.resolution)

    run_prediction(
        model=model,
        configuration=configuration,
        transforms=transforms,
        device=device,
        output_dir=predictions_directory,
    )


if __name__ == "__main__":
    main()
