import torch

from pathlib import Path

from evaluation.deployment.export_model import export_to_onnx
from evaluation.utils.configuration import load_config

from unet.build_model import build_model


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent.parent
    exports_directory = root_directory / "exports"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_config(
        current_directory=root_directory,
        configuration_path="unet/configs/inference.yaml"
    )

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        compile_model=False,
        device=str(device),
        mode="eval"
    )

    # Export the model
    model_name = checkpoint_path.stem
    export_to_onnx(
        model=model,
        device=device,
        directory=exports_directory,
        model_name=model_name,
        input_shape=(1, 3, configuration.scratch.resolution, configuration.scratch.resolution)
    )


if __name__ == "__main__":
    main()
