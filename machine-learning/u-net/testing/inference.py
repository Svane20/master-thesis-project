import torch

from tqdm.auto import tqdm

from metrics.DICE import calculate_DICE, calculate_DICE_edge


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
    model.eval()

    total_dice, total_dice_edge = 0, 0
    num_batches = 0

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            num_batches += 1

            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()

            total_dice += calculate_DICE(preds, y)
            total_dice_edge += calculate_DICE_edge(preds, y)

    # Compute average metrics
    avg_dice = total_dice / num_batches
    avg_dice_edge = total_dice_edge / num_batches

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Dice Edge Score: {avg_dice_edge:.4f}")
