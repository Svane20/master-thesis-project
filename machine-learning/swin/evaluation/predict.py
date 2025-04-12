from pathlib import Path

from libs.configuration.configuration import ConfigurationMode
from libs.evaluation.predict import run_prediction
from libs.utils.device import get_device

from swin.build_model import build_model_for_evaluation
from swin.utils.config import load_config_and_checkpoint_path


def main() -> None:
    # Directories
    current_directory = Path(__file__).resolve().parent
    predictions_directory = current_directory / "predictions"
    predictions_directory.mkdir(parents=True, exist_ok=True)

    # Load the configuration and checkpoint path
    configuration, checkpoint_path = load_config_and_checkpoint_path(ConfigurationMode.Evaluation)

    # Load the model
    device = get_device()
    model = build_model_for_evaluation(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Run the prediction
    run_prediction(model=model, configuration=configuration, device=device, output_dir=predictions_directory)


if __name__ == "__main__":
    main()
