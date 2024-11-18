import torch

import wandb
from tqdm.auto import tqdm
from typing import Optional, Dict
import time

from configuration.weights_and_biases import WeightAndBiasesConfig
from metrics.DICE import calculate_DICE, calculate_DICE_edge
from utils.checkpoints import save_checkpoint


def train(
        run: wandb.sdk.wandb_run.Run,
        configuration: WeightAndBiasesConfig,
        model: torch.nn.Module,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        device: torch.device,
        scheduler: torch.optim.lr_scheduler,
        disable_progress_bar: bool = False,
        early_stop_patience: int = 5,
        start_epoch: int = 0,
) -> None:
    """
    Trains and evaluates a model

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Args:
        run (wandb.sdk.wandb_run.Run): Weights & Biases run object
        configuration (WeightAndBiasesConfig): Configuration for Weights & Biases
        model (torch.nn.Module): Model to train and evaluate
        train_data_loader (torch.utils.data.DataLoader): Data loader for training
        test_data_loader (torch.utils.data.DataLoader): Data loader for testing
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scaler (Optional[torch.amp.GradScaler]): Gradient scaler
        device (torch.device): Device to run the training on
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        disable_progress_bar (bool): Disable tqdm progress bar. Default is False
        early_stop_patience (int): Number of epochs to wait for improvement before stopping. Default is 5
        start_epoch (int): Epoch to start from. Default is 0
    """
    # Start Weights & Biases run
    run.watch(model, optimizer, log="all", log_freq=10)

    best_val_dice, early_stop_counter = 0.0, 0
    num_epochs = start_epoch + configuration.epochs
    training_start_time = time.time()

    for epoch in tqdm(range(start_epoch, num_epochs), disable=disable_progress_bar):
        start_epoch_time = time.time()

        current_epoch = epoch + 1

        # Train step
        train_metrics = _train_one_epoch(
            model=model,
            dataloader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=current_epoch,
            num_epochs=num_epochs,
            device=device,
            disable_progress_bar=disable_progress_bar
        )

        # Test step
        test_metrics = _test_one_epoch(
            model=model,
            dataloader=test_data_loader,
            criterion=criterion,
            epoch=current_epoch,
            num_epochs=num_epochs,
            device=device,
            disable_progress_bar=disable_progress_bar
        )

        # Update learning rate - Ensure correct scheduler usage
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_metrics["dice_edge"])
        else:
            scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate epoch duration
        epoch_duration = time.time() - start_epoch_time

        # Log metrics to Weights & Biases
        run.log({
            "epoch": current_epoch,
            "learning_rate": current_lr,
            "epoch_duration": epoch_duration,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"test/{k}": v for k, v in test_metrics.items()}
        })

        print(
            f"Epoch: {current_epoch}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f} | Train Dice Edge: {train_metrics['dice_edge']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | Test Dice: {test_metrics['dice']:.4f} | Test Dice Edge: {test_metrics['dice_edge']:.4f} | "
            f"LR: {current_lr:.6f} | Epoch Duration: {epoch_duration:.2f}s"
        )

        # Checkpointing - Save the best model
        test_dice_edge = test_metrics["dice_edge"]
        if test_dice_edge > best_val_dice:
            best_val_dice = test_dice_edge

            try:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={"epoch": current_epoch, **test_metrics},
                    model_name=configuration.name_of_model,
                )
            except Exception as e:
                print(f"An error occurred during checkpointing: {e}")

            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {current_epoch}")
            break

    # Total Training Time
    total_training_time = time.time() - training_start_time
    wandb.log({"training_time": total_training_time / 60})
    print(f"[INFO] Total training time: {total_training_time / 60:.2f} minutes")


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
) -> Dict[str, float]:
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
        Dict[str, float]: Dictionary containing average loss, DICE, DICE for edges
    """
    model.train()

    total_loss = 0.0
    num_batches = 0
    total_dice, total_dice_edge = 0.0, 0.0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    for batch_idx, (X, y) in progress_bar:
        num_batches += 1

        X, y = X.to(device), y.to(device)

        # Mixed Precision Training
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
            # Forward pass
            y_pred = model(X)

            # Calculate loss
            loss = criterion(y_pred, y)

        # Zero gradients
        optimizer.zero_grad()

        if scaler is not None:
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

        total_loss += loss.item()

        # Calculate predictions
        preds = torch.sigmoid(y_pred)
        preds = (preds > 0.5).float()

        # Calculate metrics
        total_dice += calculate_DICE(preds, y)
        total_dice_edge += calculate_DICE_edge(preds, y)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": total_loss / num_batches,
                "train_dice": total_dice / num_batches,
                "train_dice_edge": total_dice_edge / num_batches
            }
        )

    # Compute average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "dice_edge": total_dice_edge / num_batches
    }

    return metrics


def _test_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
) -> Dict[str, float]:
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
        Dict[str, float]: Dictionary containing average loss, DICE, DICE for edges
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    total_dice, total_dice_edge = 0.0, 0.0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    with torch.inference_mode():
        for batch_idx, (X, y) in progress_bar:
            num_batches += 1

            X, y = X.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                # Forward pass
                y_pred = model(X)

                # Calculate loss
                loss = criterion(y_pred, y)

            total_loss += loss.item()

            # Calculate predictions
            preds = torch.sigmoid(y_pred)
            preds = (preds > 0.5).float()

            # Calculate metrics
            total_dice += calculate_DICE(preds, y)
            total_dice_edge += calculate_DICE_edge(preds, y)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "test_loss": total_loss / num_batches,
                    "test_dice": total_dice / num_batches,
                    "test_dice_edge": total_dice_edge / num_batches
                }
            )

    # Compute average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "dice_edge": total_dice_edge / num_batches
    }

    return metrics
