import torch

import wandb
from tqdm.auto import tqdm
from typing import Tuple
import time

from utils import save_checkpoint


def train(
        model: torch.nn.Module,
        model_name: str,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        epochs: int,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler = None,
        disable_progress_bar: bool = False
) -> None:
    """
    Trains and evaluates a model

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Args:
        model (torch.nn.Module): Model to train and evaluate
        model_name (str): Name of the model
        train_dataloader (torch.utils.data.DataLoader): Data loader for training
        test_dataloader (torch.utils.data.DataLoader): Data loader for testing
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scaler (torch.amp.GradScaler): Gradient scaler
        epochs (int): Number of epochs to train for
        device (torch.device): Device to run the training on
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler. Default is None.
        disable_progress_bar (bool): Disable tqdm progress bar. Default is False
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
            "model_name": model_name,
            "device": device.type
        }
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 5  # Stop if no improvement in 5 epochs

    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        start_time = time.time()

        # Train step
        train_loss, train_acc = _train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            num_epochs=epochs,
            device=device,
            disable_progress_bar=disable_progress_bar
        )

        # Test step
        test_loss, test_acc = eval_step(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            epoch=epoch,
            num_epochs=epochs,
            device=device,
            disable_progress_bar=disable_progress_bar
        )

        # Update learning rate
        if scheduler is not None:
            scheduler.step(test_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate epoch duration
        epoch_duration = time.time() - start_time

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "learning_rate": current_lr,
            "epoch_duration": epoch_duration
        })

        print(
            f"Epoch: {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"epoch_duration: {epoch_duration:.4f} "
        )

        # Checkpointing - Save the best model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            save_checkpoint(
                model=model,
                model_name=model_name,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=test_loss,
                accuracy=test_acc,
            )
            early_stop_counter = 0  # Reset counter if improvement is found
        else:
            early_stop_counter += 1

        # Early stopping condition
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


def eval_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
) -> Tuple[float, float]:
    """
    Evaluates a model for a single epoch

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Data loader
        criterion (torch.nn.Module): Loss function
        epoch (int): Current epoch
        num_epochs (int): Total number of epochs
        device (torch.device): Device to run the evaluation on
        disable_progress_bar (bool): Disable tqdm progress bar. Default is False

    Returns:
        Tuple[float, float]: Average loss and accuracy values
    """
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Testing Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    with torch.inference_mode():
        for X, y in progress_bar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            with torch.amp.autocast(device_type=device.type):
                # Forward pass
                test_pred_logits = model(X)

                # Calculate loss
                loss = criterion(test_pred_logits, y)

                # Accumulate loss and accuracy for each batch
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        # Adjust the loss and accuracy values
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def _train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
) -> Tuple[float, float]:
    """
    Trains a model for a single epoch

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scaler (torch.amp.GradScaler): Gradient scaler
        epoch (int): Current epoch
        num_epochs (int): Total number of epochs
        device (torch.device): Device to run the training on
        disable_progress_bar (bool): Disable tqdm progress bar. Default is False

    Returns:
        Tuple[float, float]: Average loss and accuracy values
    """
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    for batch, (X, y) in progress_bar:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.amp.autocast(device_type=device.type):
            # Forward pass
            y_pred = model(X)

            # Calculate loss
            loss = criterion(y_pred, y)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss and accuracy for each batch
        train_loss += loss.item()
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1),
            }
        )

    # Adjust the loss and accuracy values
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc
