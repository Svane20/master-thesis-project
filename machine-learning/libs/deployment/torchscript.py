import torch

from pathlib import Path
import logging


def export_to_torch_script(
        model: torch.nn.Module,
        model_name: str,
        directory: Path,
        dummy_input: torch.Tensor,
) -> None:
    """
    Export the model to TorchScript format.

    Args:
        model (torch.nn.Module): Model to export.
        model_name (str): Name of the model.
        directory (Path): Directory to save the TorchScript model to.
        dummy_input (torch.Tensor): Dummy input tensor.
    """
    # Create export directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / f"{model_name}_ts.pt"

    # Set model to evaluation mode
    model.eval()

    # Export model to TorchScript
    try:
        scripted_model = torch.jit.script(model, dummy_input)
        scripted_model.save(save_path)

        logging.info(f"TorchScript model exported at {save_path}")
    except Exception as e:
        raise RuntimeError(f"TorchScript export failed: {e}")

    print(f"Model exported to TorchScript at {save_path}")
