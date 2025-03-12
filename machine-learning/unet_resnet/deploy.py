import torch

from pathlib import Path
import platform

from libs.configuration.configuration import load_configuration_and_checkpoint
from libs.deployment.onnx import export_to_onnx
from libs.deployment.minimal import export_model
from libs.deployment.torchscript import export_to_torch_script

from build_model import build_model


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent

    # Get configuration based on OS
    if platform.system() == "Windows":
        configuration_path: Path = root_directory / "unet_resnet/configs/deployment_windows.yaml"
    else:  # Assume Linux for any non-Windows OS
        configuration_path: Path = root_directory / "unet_resnet/configs/deployment_linux.yaml"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_configuration_and_checkpoint(configuration_path, is_deployment=True)
    destination_directory = Path(configuration.deployment.destination_directory)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        configuration=configuration.model,
        checkpoint_path=checkpoint_path,
        compile_model=False,
        device=str(device),
        mode="eval"
    )
    dummy_input = torch.randn(1, 3, configuration.scratch.resolution, configuration.scratch.resolution).to(device)

    # Export the minimal model checkpoint
    model_name = checkpoint_path.stem
    export_model(
        model=model,
        device=device,
        directory=destination_directory,
        model_name=model_name,
    )

    # Export the model to TorchScript
    export_to_torch_script(
        model=model,
        directory=destination_directory,
        model_name=model_name,
        dummy_input=dummy_input,
    )

    # Export the model to ONNX
    export_to_onnx(
        model=model,
        directory=destination_directory,
        model_name=model_name,
        dummy_input=dummy_input,
    )

    print("Production build completed successfully.")


if __name__ == "__main__":
    main()
