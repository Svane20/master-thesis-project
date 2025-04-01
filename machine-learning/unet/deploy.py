from libs.configuration.configuration import ConfigurationMode
from libs.deployment.deploy import deploy_model
from libs.utils.device import get_device

from unet.build_model import build_model_for_deployment
from unet.utils.config import load_config_and_checkpoint_path


def main() -> None:
    configuration, checkpoint_path = load_config_and_checkpoint_path(ConfigurationMode.Deployment)

    # Load the model
    device = get_device()
    model = build_model_for_deployment(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Export the model
    deploy_model(
        model=model,
        model_name=checkpoint_path.stem,
        device=device,
        configuration=configuration
    )


if __name__ == "__main__":
    main()
