import torch

import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler = None,
) -> Dict[str, List[float]]:
    """
    Trains and evaluates a model

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Args:
        model (torch.nn.Module): Model to train and evaluate
        train_dataloader (torch.utils.data.DataLoader): Data loader for training
        test_dataloader (torch.utils.data.DataLoader): Data loader for testing
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        epochs (int): Number of epochs to train for
        device (torch.device): Device to run the training on
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler. Default is None.
    """
    # Initialize Weights & Biases
    wandb.init(
        project="MNIST",
        config={
            "epochs": epochs,
            "batch_size": train_dataloader.batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dataset": "MNIST",
            "architecture": "CNN",
            "model_name": model.__class__.__name__,
            "device": device.type
        }
    )

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        # Train step
        train_loss, train_acc = _train_step(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
        )

        # Test step
        test_loss, test_acc = eval_step(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device
        )

        # Log metrics to Weights & Biases
        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        })

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def eval_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates a model for a single epoch

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run the evaluation on

    Returns:
        Tuple[float, float]: Average loss and accuracy values
    """
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate loss
            loss = criterion(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust the loss and accuracy values
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def _train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler = None,
) -> Tuple[float, float]:
    """
    Trains a model for a single epoch

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler. Default is None.
        device (torch.device): Device to run the training on

    Returns:
        Tuple[float, float]: Average loss and accuracy values
    """
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    for _, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate loss
        loss = criterion(y_pred, y)
        train_loss += loss.item()

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    if scheduler is not None:
        scheduler.step()

    # Adjust the loss and accuracy values
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc
