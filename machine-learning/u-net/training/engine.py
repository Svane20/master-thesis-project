import torch

import wandb
from tqdm.auto import tqdm
from typing import Optional
import time
from contextlib import nullcontext

from utils import save_checkpoint


def train(
        model: torch.nn.Module,
        model_name: str,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
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
        train_data_loader (torch.utils.data.DataLoader): Data loader for training
        test_data_loader (torch.utils.data.DataLoader): Data loader for testing
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scaler (Optional[torch.amp.GradScaler]): Gradient scaler
        epochs (int): Number of epochs to train for
        device (torch.device): Device to run the training on
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler. Default is None.
        disable_progress_bar (bool): Disable tqdm progress bar. Default is False
    """
    # Initialize Weights & Biases
    wandb.init(
        project="U-NET",
        config={
            "epochs": epochs,
            "batch_size": train_data_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dataset": "ADE20K",
            "architecture": "U-NET",
            "model_name": model_name,
            "device": device.type
        }
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 5  # Stop if no improvement in 5 epochs

    # Track training time
    training_start_time = time.time()

    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        start_epoch_time = time.time()

        # Train step
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            num_epochs=epochs,
            device=device,
            disable_progress_bar=disable_progress_bar
        )

        # Test step
        test_loss = _test_one_epoch(
            model=model,
            dataloader=test_data_loader,
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
        epoch_duration = time.time() - start_epoch_time

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "test/loss": test_loss,
            "learning_rate": current_lr,
            "epoch_duration": epoch_duration
        })

        print(
            f"Epoch: {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
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
            )
            early_stop_counter = 0  # Reset counter if improvement is found
        else:
            early_stop_counter += 1

        # Early stopping condition
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Calculate total training time
    total_training_time = time.time() - training_start_time

    # Print total training time
    total_training_time_minutes = total_training_time / 60
    print(f"[INFO] Total training time: {total_training_time_minutes:.2f} minutes")

    # Finish Weights & Biases run
    wandb.finish()


def _train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
) -> float:
    """
    Trains a model for a single epoch

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scaler (Optional[torch.amp.GradScaler]): Gradient scaler
        epoch (int): Current epoch
        num_epochs (int): Total number of epochs
        device (torch.device): Device to run the training on
        disable_progress_bar (bool): Disable tqdm progress bar. Default is False

    Returns:
        float: Average loss values
    """
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    for batch, (X, y) in progress_bar:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext():
            # Forward pass
            y_pred = model(X)

            # Calculate loss
            loss = criterion(y_pred, y)

        # Zero gradients
        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            # Backward pass
            scaler.scale(loss).backward()

            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

        # Accumulate loss for each batch
        train_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
            }
        )

    # Adjust the loss and accuracy values
    train_loss /= len(dataloader)

    return train_loss


def _test_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
) -> float:
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
        float: Average loss values
    """
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    with torch.inference_mode():
        for batch, (X, y) in progress_bar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            with torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext():
                # Forward pass
                test_pred_logits = model(X)

                # Calculate loss
                loss = criterion(test_pred_logits, y)

            # Accumulate loss and accuracy for each batch
            test_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "test_loss": test_loss / (batch + 1),
                }
            )

        # Adjust the loss and accuracy values
        test_loss /= len(dataloader)

    return test_loss
