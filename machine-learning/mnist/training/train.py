import torch
import torch.cuda.amp
from torch import nn
from torchvision.transforms import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

import argparse
import sys

from utils import set_seeds, get_model_summary, get_device
from model.mnist import FashionMnistModelV0
from training import engine
from dataset.data_loader import create_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FashionMNIST model.")
    parser.add_argument("--model_name", type=str, default="FashionMNISTModelV0", help="Model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training", choices=range(1, 129))
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", choices=[0.001, 0.01, 0.1, 0.0001])
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training", choices=range(1, 100))
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


def main():
    # Parse command-line arguments
    args = parse_args()

    # Create data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1-channel grayscale
    ])
    train_dataloader, test_dataloader, class_names = create_data_loaders(
        batch_size=args.batch_size,
        transform=transform
    )

    # Set random seed
    if args.seed is not None:
        set_seeds(seed=args.seed)

    # Set CuDNN benchmark and deterministic
    torch.backends.cudnn.benchmark = True  # Enable if input sizes are constant
    torch.backends.cudnn.deterministic = False  # Set False for better performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for mixed precision

    # Setup device
    device = get_device()

    # Instantiate the model
    model = FashionMnistModelV0(
        input_shape=1,
        hidden_units=32,
        output_shape=len(class_names)
    ).to(device)

    # Summary of model
    if args.show_summary:
        get_model_summary(model, input_size=(32, 1, 28, 28))

    # Setup loss function, optimizer, lr scheduler and gradient scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = torch.amp.GradScaler()  # Mixed Precision Training

    # Train the model
    engine.train(
        model=model,
        model_name=args.model_name,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=args.epochs,
        scheduler=scheduler,
        disable_progress_bar=True
    )


if __name__ == "__main__":
    main()
