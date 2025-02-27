import torch

from pathlib import Path
import platform

from configuration.configuration import load_configuration_and_checkpoint
from deployment.export_model import export_to_onnx
from deployment.trim_checkpoint import save_model_checkpoint
from unet.build_model import build_unet_model


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent

    # Get configuration based on OS
    if platform.system() == "Windows":
        configuration_path: Path = root_directory / "unet/configuration/deployment_windows.yaml"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = root_directory / "unet/configuration/deployment_linux.yaml"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_configuration_and_checkpoint(configuration_path, is_deployment=True)
    exports_directory = Path(configuration.deployment.destination_directory)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_model(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        compile_model=False,
        device=str(device),
        mode="eval"
    )

    # Export the minimal model checkpoint
    model_name = checkpoint_path.stem
    save_model_checkpoint(
        model=model,
        device=device,
        directory=exports_directory,
        model_name=model_name,
    )

    # Export the model to ONNX
    export_to_onnx(
        model=model,
        device=device,
        directory=exports_directory,
        model_name=model_name,
        input_shape=(1, 3, configuration.scratch.resolution, configuration.scratch.resolution)
    )

    print("Production build completed successfully.")


if __name__ == "__main__":
    main()
