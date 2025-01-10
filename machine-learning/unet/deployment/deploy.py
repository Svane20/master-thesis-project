import torch

from pathlib import Path

from deployment.export_model import export_to_onnx
from deployment.trim_checkpoint import save_model_checkpoint
from evaluation.utils.configuration import load_config
from unet.build_model import build_unet_model


def main() -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent
    exports_directory = root_directory / "exports"

    # Load configuration and checkpoint
    configuration, checkpoint_path = load_config(
        current_directory=root_directory,
        configuration_path="unet/configuration/deployment.yaml"
    )

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
        model_name=f"{model_name}_production",
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
