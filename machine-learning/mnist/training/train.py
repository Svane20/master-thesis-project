from torch import nn
from torchvision.transforms import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

import argparse
import sys

from utils import set_seeds, save_model, get_model_summary, get_device
from model.mnist import FashionMnistModelV0
from training import engine
from dataset.data_loader import create_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FashionMNIST model.")
    parser.add_argument("--model_name", type=str, default="FashionMNISTModelV0", help="Model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
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

    # Setup loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    engine.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=args.epochs,
        scheduler=scheduler,
    )

    # Save the model
    save_model(model=model, model_name=args.model_name)


if __name__ == "__main__":
    main()
