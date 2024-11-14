import torch
import torch.cuda.amp
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from typing import Tuple
import argparse
import sys
import warnings
import os
from pathlib import Path

from constants.directories import DATA_TRAIN_DIRECTORY, DATA_TEST_DIRECTORY
from constants.hyperparameters import BATCH_SIZE, SEED, LEARNING_RATE, NUM_EPOCHS
from constants.outputs import MODEL_NAME
from dataset.data_loaders import create_data_loaders
from dataset.transforms import get_train_transforms, get_test_transforms
from model.unet import UNetV0
from training import engine
from training import custom_criterions
from utils import set_seeds, get_device, get_model_summary, load_checkpoint

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a U-Net model.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
        choices=range(1, 129)
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
        choices=[0.0001, 0.001, 0.01, 0.1]
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of epochs for training",
        choices=range(1, 100)
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--show-summary", type=bool, default=False, help="Show the summary of the model")
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


def clear_cuda_cache() -> None:
    """
    Clear the CUDA cache.
    """
    torch.cuda.empty_cache()


def setup_torch_backend() -> None:
    """
    Set up the torch backend for training.
    """
    if torch.cuda.is_available():
        # Mixed Precision Training if available
        torch.backends.cuda.matmul.allow_tf32 = True if torch.cuda.get_device_capability() >= (8, 0) else False
        torch.backends.cudnn.benchmark = True  # Enable if input sizes are constant
        torch.backends.cudnn.deterministic = False  # Set False for better performance
    else:
        warnings.warn("GPU is not available. Running on CPU.")


def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Get the training and test data loaders.

    Args:
        batch_size (int): Batch size for the data loaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders.
    """
    transform = get_train_transforms()
    target_transform = get_test_transforms()

    return create_data_loaders(
        train_directory=DATA_TRAIN_DIRECTORY,
        test_directory=DATA_TEST_DIRECTORY,
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=os.cpu_count() if torch.cuda.is_available() else 2,
    )


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        set_seeds(seed=args.seed)

    # Setup device and torch backend
    device = get_device()
    setup_torch_backend()

    # Clear the CUDA cache
    if device.type == "cuda":
        clear_cuda_cache()

    # Instantiate the model
    model = UNetV0(in_channels=3, out_channels=1, dropout=0.5).to(device)

    # Print the model summary
    if args.show_summary:
        get_model_summary(model, input_size=(args.batch_size, 3, 224, 224))

    # Setup loss function, optimizer, lr scheduler and gradient scaler (Mixed Precision)
    criterion = custom_criterions.EdgeWeightedBCEDiceLoss(edge_weight=5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # Load checkpoint if specified
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model, optimizer, scheduler, checkpoint_info = load_checkpoint(
            model=model,
            model_name=args.model_name,
            device=device,
            directory=Path(args.checkpoint_path).parent,
            optimizer=optimizer,
            scheduler=scheduler,
            is_eval=False
        )
        start_epoch = checkpoint_info["epoch"] + 1  # Continue from the next epoch
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Prepare data loaders
    train_data_loader, test_data_loader = get_data_loaders(batch_size=args.batch_size)

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
