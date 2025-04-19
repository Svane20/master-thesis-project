import torch

from libs.configuration.configuration import ConfigurationMode
from libs.deployment.deploy import deploy_model
from libs.utils.device import get_devices_for_deployment

from resnet.build_model import build_model_for_deployment
from resnet.utils.config import load_config_and_checkpoint_path


def main() -> None:
    configuration, checkpoint_path = load_config_and_checkpoint_path(ConfigurationMode.Deployment)

    # Load the devices for deployment
    devices = get_devices_for_deployment(configuration.deployment)

    for device in devices:
        # Load the model
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

        # Clear the cache
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
