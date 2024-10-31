import torch

from pathlib import Path

def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Random seed to set. Default is 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(
        model: torch.nn.Module,
        model_name: str,
        directory: str = "models",
        extension: str = "pth"
) -> None:
    """
    Saves a model to the specified directory.

    Args:
        model (torch.nn.Module): Model to save.
        model_name (str): Name of the model.
        directory (str): Directory to save the model to. Default is "models".
        extension (str): Extension to use. Default is ".pth".
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    assert extension in ["pth", "pt"], "Extension must be either 'pth' or 'pt'"
    save_path = dir_path / f"{model_name}.{extension}"

    print(f"[INFO] Saving model to {save_path}")
    torch.save(obj=model.state_dict(), f=save_path)
