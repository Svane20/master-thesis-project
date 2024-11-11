import torch

import wandb
from tqdm.auto import tqdm
from typing import Optional, Tuple, Any
import time
from contextlib import nullcontext

from metrics.DICE import calculate_DICE
from metrics.IoU import calculate_IoU
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

    best_val_dice = 0.0
    early_stop_counter = 0
    early_stop_patience = 5  # Stop if no improvement in 5 epochs

    # Track training time
    training_start_time = time.time()

    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        start_epoch_time = time.time()

        # Train step
        train_loss, train_iou, train_dice = _train_one_epoch(
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
        test_loss, test_iou, test_dice = _test_one_epoch(
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
            scheduler.step(test_dice)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate epoch duration
        epoch_duration = time.time() - start_epoch_time

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/IoU": train_iou,
            "train/Dice": train_dice,
            "test/loss": test_loss,
            "test/IoU": test_iou,
            "test/Dice": test_dice,
            "learning_rate": current_lr,
            "epoch_duration": epoch_duration
        })

        print(
            f"Epoch: {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Dice: {train_dice:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test IoU: {test_iou:.4f} | Test Dice: {test_dice:.4f} | "
            f"LR: {current_lr:.6f} | Epoch Duration: {epoch_duration:.2f}s"
        )

        # Checkpointing - Save the best model
        if test_dice > best_val_dice:
            best_val_dice = test_dice
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
) -> Tuple[float, float, float]:
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
        Tuple[float, float, float]: Average loss, IoU and DICE values
    """
    model.train()

    # Setup training loss and metrics values
    train_loss = 0
    train_iou = 0
    train_dice = 0
    num_batches = 0

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

        # Calculate metrics
        with torch.inference_mode():
            preds = torch.sigmoid(y_pred)
            preds = (preds > 0.5).float()
            batch_iou = calculate_IoU(preds, y)
            batch_dice = calculate_DICE(preds, y)

        train_iou += batch_iou
        train_dice += batch_dice
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_iou": train_iou / num_batches,
                "train_dice": train_dice / num_batches,
            }
        )

    # Compute average metrics
    train_loss /= num_batches
    train_iou /= num_batches
    train_dice /= num_batches

    return train_loss, train_iou, train_dice


def _test_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
) -> Tuple[float, float, float]:
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
        Tuple[float, float, float]: Average loss, IoU and DICE values
    """
    model.eval()

    # Setup test loss and metrics values
    test_loss = 0
    test_iou = 0
    test_dice = 0
    num_batches = 0

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

            # Calculate metrics
            preds = torch.sigmoid(test_pred_logits)
            preds = (preds > 0.5).float()
            batch_iou = calculate_IoU(preds, y)
            batch_dice = calculate_DICE(preds, y)

            test_iou += batch_iou
            test_dice += batch_dice
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "test_loss": test_loss / (batch + 1),
                    "test_iou": test_iou / num_batches,
                    "test_dice": test_dice / num_batches,
                }
            )

    # Compute average metrics
    test_loss /= num_batches
    test_iou /= num_batches
    test_dice /= num_batches

    return test_loss, test_iou, test_dice
