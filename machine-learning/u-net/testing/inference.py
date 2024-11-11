import torch

from tqdm.auto import tqdm

from metrics.DICE import calculate_DICE
from metrics.IoU import calculate_IoU
from metrics.precision import calculate_precision
from metrics.recall import calculate_recall


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
    # Initialize variables
    num_correct = 0
    num_pixels = 0

    # Initialize variables for metrics
    total_dice = 0
    total_iou = 0
    total_precision = 0.0
    total_recall = 0.0

    # Initialize variables for counting
    num_batches = 0

    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()

            # Calculate metrics
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            total_dice += calculate_DICE(preds, y)
            total_iou += calculate_IoU(preds, y)
            total_precision += calculate_precision(preds, y)
            total_recall += calculate_recall(preds, y)

            num_batches += 1

    # Compute average metrics
    accuracy = num_correct / num_pixels * 100
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")
