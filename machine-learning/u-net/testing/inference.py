import torch
from torchvision.utils import save_image

from tqdm.auto import tqdm
from pathlib import Path

from constants.directories import OUTPUT_DIRECTORY


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> None:
    """
    Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)

            preds = (y_preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct} / {num_pixels} with accuracy {num_correct / num_pixels * 100:.2f}")

    print(f"Dice score: {dice_score / len(data_loader)}")


def save_predictions_as_images(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        directory: Path = OUTPUT_DIRECTORY,
) -> None:
    """
    Save the model predictions as images.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
        directory (Path): Directory to save the images to. Default is "output".
    """
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(data_loader), desc="Saving predictions"):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)

            preds = (y_preds > 0.5).float()

            # Create directory if it does not exist
            directory.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for predictions and targets
            prediction_directory = directory / "predictions"
            target_directory = directory / "targets"
            prediction_directory.mkdir(parents=True, exist_ok=True)
            target_directory.mkdir(parents=True, exist_ok=True)

            prediction_path = prediction_directory / f"prediction_{batch}.png"
            target_path = target_directory / f"target_{batch}.png"

            # Save the predictions and target images
            save_image(preds, prediction_path)
            save_image(y, target_path)

            print(f"Saved prediction at {prediction_path}")
            print(f"Saved target at {target_path}")
