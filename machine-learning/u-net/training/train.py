import torch
import torch.cuda.amp
from torch import optim
from torch.optim import lr_scheduler

import argparse
import sys
import os
from pathlib import Path

from constants.directories import DATA_TRAIN_DIRECTORY, DATA_TEST_DIRECTORY
from constants.hyperparameters import BATCH_SIZE, SEED, LEARNING_RATE, NUM_EPOCHS, LEARNING_RATE_DECAY
from constants.outputs import MODEL_OUTPUT_NAME, TRAINED_MODEL_CHECKPOINT_NAME
from dataset.data_loaders import create_data_loaders
from dataset.transforms import get_train_transforms, get_test_transforms
from model.unet import UNetV0
from training import engine
from training import custom_criterions
from utils.checkpoints import load_checkpoint
from utils.device import get_device
from utils.seeds import set_seeds
from utils.training import setup_AMP

# Prevent unwanted updates in Albumentations
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a U-Net model.")
    parser.add_argument("--model_name", type=str, default=MODEL_OUTPUT_NAME, help="Model name")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=LEARNING_RATE_DECAY, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs for training")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint for fine-tuning")

    args = parser.parse_args()

    if args.batch_size <= 0:
        print("Error: batch_size must be a positive integer.")
        sys.exit(1)

    if args.lr <= 0:
        print("Error: learning rate must be a positive float.")
        sys.exit(1)

    if args.epochs <= 0:
        print("Error: epochs must be a positive integer.")
        sys.exit(1)

    return args


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    # Set random seed
    set_seeds(seed=args.seed)

    # Setup device and torch backend
    device = get_device()
    setup_AMP()

    # Clear the CUDA cache
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Instantiate Model
    model = UNetV0(in_channels=3, out_channels=1, dropout=0.5).to(device)

    # Loss, Optimizer, Scheduler, AMP
    criterion = custom_criterions.EdgeWeightedBCEDiceLoss(edge_weight=5, edge_loss_weight=1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    # Load Checkpoint
    start_epoch = 0
    if args.checkpoint_path:
        model, optimizer, scheduler, checkpoint_info = load_checkpoint(
            model=model,
            model_name=TRAINED_MODEL_CHECKPOINT_NAME,
            device=device,
            directory=Path(args.checkpoint_path),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = checkpoint_info["epoch"] + 1

    # Prepare data loaders
    transform = get_train_transforms()
    target_transform = get_test_transforms()
    train_data_loader, test_data_loader = create_data_loaders(
        train_directory=DATA_TRAIN_DIRECTORY,
        test_directory=DATA_TEST_DIRECTORY,
        batch_size=args.batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=os.cpu_count() - 1,
    )

    # Train the model
    engine.train(
        model=model,
        model_name=args.model_name,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        device=device,
        epochs=args.epochs,
        scheduler=scheduler,
        start_epoch=start_epoch
    )


if __name__ == "__main__":
    main()
