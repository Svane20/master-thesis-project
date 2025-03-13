from libs.configuration.configuration import ConfigurationMode
from libs.evaluation.eval import run_evaluation
from libs.utils.device import get_device

from unet.build_model import build_model_for_evaluation
from unet.utils import load_config_and_checkpoint_path


def main() -> None:
    configuration, checkpoint_path = load_config_and_checkpoint_path(ConfigurationMode.Evaluation)

    # Load the model
    device = get_device()
    model = build_model_for_evaluation(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Run evaluation
    run_evaluation(
        model=model,
        configuration=configuration,
        device=device
    )


if __name__ == "__main__":
    main()
