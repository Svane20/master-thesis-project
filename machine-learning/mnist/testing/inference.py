import torch

from tqdm.auto import tqdm


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> torch.Tensor:
    """
    Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation

    Returns:
        torch.Tensor: Predictions
    """
    y_preds = []

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

            y_preds.append(y_pred.cpu())

    return torch.cat(y_preds)
