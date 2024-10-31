import torch

import argparse
import sys

import data_setup, engine, model_builder, utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FashionMNIST model.")
    parser.add_argument("--model_name", type=str, default="FashionMNISTModelV0", help="Model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.batch_size <= 0:
        print("Error: batch_size must be a positive integer.")
        sys.exit(1)

    if args.lr <= 0:
        print("Error: learning rate must be a positive float.")
        sys.exit(1)

    # Validate number of epochs (e.g., must be positive)
    if args.epochs <= 0:
        print("Error: num_epochs must be a positive integer.")
        sys.exit(1)

    return args


def main():
    # Parse command-line arguments
    args = parse_args()

    # Create data loaders
    train_dataloader, test_dataloader, class_names = data_setup.create_data_loaders(batch_size=args.batch_size)

    # Set random seed
    if args.seed is not None:
        utils.set_seeds(seed=args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_builder.FashionMnistModelV0(
        input_shape=1,
        hidden_units=10,
        output_shape=len(class_names)
    ).to(device)

    # Setup loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    engine.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=args.epochs
    )

    # Save the model
    utils.save_model(model=model, model_name=args.model_name)


if __name__ == "__main__":
    main()
