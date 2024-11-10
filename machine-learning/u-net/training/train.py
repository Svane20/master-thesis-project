import torch
import torch.cuda.amp
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader

from typing import Tuple
import argparse
import sys
import warnings
import os

from constants.directories import DATA_TRAIN_DIRECTORY, DATA_TEST_DIRECTORY
from dataset.data_loader import create_data_loaders
from model.unet import UNetV0
from training import engine
from utils import set_seeds, get_device, get_model_summary


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a U-Net model.")
    parser.add_argument("--model_name", type=str, default="UNetV0", help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training", choices=range(1, 129))
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", choices=[0.0001, 0.001, 0.01, 0.1])
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training", choices=range(1, 100))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--show-summary", type=bool, default=False, help="Show the summary of the model")

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
    # Define the transformations
    train_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    train_mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    test_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    test_mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return create_data_loaders(
        train_directory=DATA_TRAIN_DIRECTORY,
        test_directory=DATA_TEST_DIRECTORY,
        batch_size=batch_size,
        transform_image=train_image_transform,
        transform_mask=train_mask_transform,
        target_transform_image=test_image_transform,
        target_transform_mask=test_mask_transform,
        num_workers=os.cpu_count() if torch.cuda.is_available() else 2,
    )


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    train_data_loader, test_data_loader = get_data_loaders(batch_size=args.batch_size)

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
    model = UNetV0(
        in_channels=3,
        out_channels=1,
    ).to(device)

    # Print the model summary
    if args.show_summary:
        get_model_summary(model, input_size=(args.batch_size, 3, 224, 224))

    # Setup loss function, optimizer, lr scheduler and gradient scaler (Mixed Precision)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = torch.amp.GradScaler() if device.type == "cuda" and torch.cuda.get_device_capability() >= (8, 0) else None

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
    )


if __name__ == "__main__":
    main()
