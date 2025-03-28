from libs.configuration.configuration import ConfigurationMode
from libs.datasets.synthetic.synthetic_dataset import DatasetPhase
from libs.evaluation.eval import run_evaluation
from libs.utils.device import get_device

from unet.build_model import build_model_for_evaluation
from unet.utils.config import load_config_and_checkpoint_path
from unet.utils.transforms import get_transforms


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

    # Create transforms
    transforms = get_transforms(phase=DatasetPhase.Test, resolution=configuration.scratch.resolution)

    # Run evaluation
    run_evaluation(model=model, device=device, configuration=configuration, transforms=transforms)


if __name__ == "__main__":
    main()
