from pathlib import Path
from libs.configuration.configuration import ConfigurationMode
from libs.evaluation.predict import run_prediction
from libs.utils.device import get_device

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

    run_prediction(
        model=model,
        configuration=configuration,
        device=device,
        output_dir=predictions_directory,
    )


if __name__ == "__main__":
    main()
