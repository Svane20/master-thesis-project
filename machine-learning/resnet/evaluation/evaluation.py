from libs.configuration.configuration import ConfigurationMode
from libs.evaluation.eval import run_evaluation
from libs.utils.device import get_device

from resnet.build_model import build_model_for_evaluation
from resnet.utils.config import load_config_and_checkpoint_path


def main() -> None:
    # Load the configuration and checkpoint path
    configuration, checkpoint_path = load_config_and_checkpoint_path(ConfigurationMode.Evaluation)

    # Load the model
    device = get_device()
    model = build_model_for_evaluation(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Run evaluation
    run_evaluation(model=model, device=device, configuration=configuration)


if __name__ == "__main__":
    main()
