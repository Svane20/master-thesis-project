import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.models.segformer.image_processing_segformer import SegformerImageProcessor

from tqdm.auto import tqdm
from typing import Optional, Dict

from configuration.weights_and_biases import WeightAndBiasesConfig
from metrics.dice import dice_coefficient
from metrics.iou import iou, boundary_iou
from training.early_stopping import EarlyStopping
import time

from utils.checkpoints import save_checkpoint


def train(
        configuration: WeightAndBiasesConfig,
        model: torch.nn.Module,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        image_processor: SegformerImageProcessor,
        dice_criterion: torch.nn.Module,
        boundary_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        device: torch.device,
        scheduler: torch.optim.lr_scheduler,
        early_stopping: EarlyStopping,
        disable_progress_bar: bool = False,
        start_epoch: int = 0,
):
    num_epochs = start_epoch + configuration.epochs
    training_start_time = time.time()

    for epoch in tqdm(range(start_epoch, num_epochs), disable=disable_progress_bar):
        start_epoch_time = time.time()

        current_epoch = epoch + 1

        # Train step
        train_metrics = _train_one_epoch(
            model=model,
            dataloader=train_data_loader,
            image_processor=image_processor,
            dice_criterion=dice_criterion,
            boundary_criterion=boundary_criterion,
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
            image_processor=image_processor,
            dice_criterion=dice_criterion,
            boundary_criterion=boundary_criterion,
            epoch=current_epoch,
            num_epochs=num_epochs,
            device=device,
            disable_progress_bar=disable_progress_bar
        )

        validation_metric = test_metrics.get("dice", None)

        # Step the scheduler
        _scheduler_step(scheduler)

        # Update early stopping
        early_stopping.step(validation_metric)

        # Save checkpoint if improved
        if early_stopping.has_improved:
            try:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                    metrics={"epoch": current_epoch, **test_metrics},
                    model_name=configuration.name_of_model,
                )
            except Exception as e:
                print(f"An error occurred during checkpointing: {e}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate epoch duration
        epoch_duration = time.time() - start_epoch_time

        # Print metrics
        train_metrics_str = " | ".join([f"Train {k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()])
        test_metrics_str = " | ".join([f"Test {k.capitalize()}: {v:.4f}" for k, v in test_metrics.items()])
        print(
            f"Epoch: {current_epoch}/{num_epochs} | LR: {current_lr:.6f} | Epoch Duration: {epoch_duration:.2f}s\n"
            f"{train_metrics_str} | {test_metrics_str}"
        )

        # Check for early stopping
        if early_stopping.should_stop:
            print(f"[INFO] Early stopping activated. Best score: {early_stopping.best_score:.4f}")
            break

    # Total Training Time
    total_training_time = time.time() - training_start_time
    print(f"[INFO] Total training time: {total_training_time / 60:.2f} minutes")


def _train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        image_processor: SegformerImageProcessor,
        dice_criterion: torch.nn.Module,
        boundary_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
):
    model.train()

    total_loss = 0.0
    num_batches = 0
    total_dice, total_iou, total_boundary_iou = 0, 0, 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    for batch_idx, (images, masks) in progress_bar:
        num_batches += 1

        # Process images
        inputs = image_processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        # Process masks
        masks = masks.unsqueeze(1).float().to(device, non_blocking=True)
        if masks.shape[-2:] != pixel_values.shape[-2:]:
            masks = F.interpolate(masks, size=pixel_values.shape[-2:], mode='nearest')

        # Mixed Precision Training
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
            # Forward pass
            outputs = model(pixel_values=pixel_values)

            # Calculate loss
            dice_loss = dice_criterion(outputs, masks)
            boundary_loss = boundary_criterion(outputs, masks)
            loss = dice_loss + boundary_loss

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

        # Calculate metrics
        total_dice += dice_coefficient(outputs, masks)
        total_iou += iou(outputs, masks)
        total_boundary_iou += boundary_iou(outputs, masks)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": total_loss / num_batches,
                "train_dice": total_dice / num_batches,
                "train_iou": total_iou / num_batches,
                "train_boundary_iou": total_boundary_iou / num_batches
            }
        )

    # Compute average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
        "boundary_iou": total_boundary_iou / num_batches
    }

    return metrics


def _test_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        image_processor: SegformerImageProcessor,
        dice_criterion: torch.nn.Module,
        boundary_criterion: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        device: torch.device,
        disable_progress_bar: bool = False
):
    model.eval()

    total_loss = 0.0
    num_batches = 0
    total_dice, total_iou, total_boundary_iou = 0, 0, 0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}/{num_epochs}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    with torch.inference_mode():
        for batch_idx, (images, masks) in progress_bar:
            num_batches += 1

            # Process images
            inputs = image_processor(images=list(images), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            # Process masks
            masks = masks.unsqueeze(1).float().to(device, non_blocking=True)
            if masks.shape[-2:] != pixel_values.shape[-2:]:
                masks = F.interpolate(masks, size=pixel_values.shape[-2:], mode='nearest')

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                # Forward pass
                outputs = model(pixel_values=pixel_values)

                # Calculate loss
                dice_loss = dice_criterion(outputs, masks)
                boundary_loss = boundary_criterion(outputs, masks)
                loss = dice_loss + boundary_loss

            total_loss += loss.item()

            # Calculate metrics
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou(outputs, masks)
            total_boundary_iou += boundary_iou(outputs, masks)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "test_loss": total_loss / num_batches,
                    "test_dice": total_dice / num_batches,
                    "test_iou": total_iou / num_batches,
                    "test_boundary_iou": total_boundary_iou / num_batches
                }
            )

    # Compute average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
        "boundary_iou": total_boundary_iou / num_batches
    }

    return metrics


def _scheduler_step(scheduler: torch.optim.lr_scheduler, validation_metric: Optional[float] = None) -> None:
    """
    Steps the scheduler based on its type and current training phase.

    Args:
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        validation_metric (Optional[float]): Validation metric for ReduceLROnPlateau.
    """
    if isinstance(scheduler, ReduceLROnPlateau):
        if validation_metric is not None:
            scheduler.step(validation_metric)
        else:
            print("[WARNING] Missing metric for ReduceLROnPlateau. Skipping scheduler step.")
    else:
        scheduler.step()
