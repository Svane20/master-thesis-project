import torch
import torch.cuda.amp
import wandb
from torch import optim
from torch.optim import lr_scheduler
from transformers import AutoImageProcessor

import argparse
import sys
import os
import traceback

from configuration.weights_and_biases import WeightAndBiasesConfig
from constants.directories import DATA_TRAIN_DIRECTORY, DATA_TEST_DIRECTORY
from constants.hyperparameters import LEARNING_RATE, LEARNING_RATE_DECAY, BATCH_SIZE, NUM_EPOCHS, SEED
from constants.outputs import MODEL_OUTPUT_NAME
from constants.seg_former import MODEL_NAME
from dataset.data_loaders import create_data_loaders
from dataset.transforms import get_train_transforms, get_test_transforms
from training import custom_criterions, engine
from training.early_stopping import EarlyStopping
from utils.device import get_device, get_torch_compile_backend
from utils.seeds import set_seeds
from utils.training import setup_AMP
from models.unet_r import UNETR

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
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint for training")
    parser.add_argument("--use_warmup", type=bool, default=False, help="Use warmup scheduling for training")

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

    # Setup device and torch backend
    device = get_device()
    setup_AMP()

    # Setup Weights & Biases
    configuration = WeightAndBiasesConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        learning_rate_decay=args.lr_decay,
        seed=args.seed,
        dataset="Carvana",
        architecture="UNET-R",
        name_of_model=args.model_name,
        device=device.type
    )

    set_seeds(seed=configuration.seed)

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Instantiate Model
    model = UNETR(out_channels=1).to(device)

    # Losses, Optimizer, Scheduler and AMP
    dice_criterion = custom_criterions.DiceLoss()
    boundary_criterion = custom_criterions.BoundaryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=configuration.learning_rate,
                            weight_decay=configuration.learning_rate_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    # Define early stopping for DICE score (max)
    early_stopping = EarlyStopping(patience=10, min_delta=0.0, verbose=True, mode="max")

    start_epoch = 0

    # Compile model for faster training
    model = torch.compile(model, backend=get_torch_compile_backend())

    # Prepare data loaders
    transform = get_train_transforms()
    target_transform = get_test_transforms()
    train_data_loader, test_data_loader = create_data_loaders(
        train_directory=DATA_TRAIN_DIRECTORY,
        test_directory=DATA_TEST_DIRECTORY,
        batch_size=configuration.batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=os.cpu_count() - 1,
        pin_memory=True
    )

    # Instantiate the image processor
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    try:
        # Train the model
        engine.train(
            configuration=configuration,
            model=model,
            dice_criterion=dice_criterion,
            boundary_criterion=boundary_criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            image_processor=image_processor,
            device=device,
            early_stopping=early_stopping,
            start_epoch=start_epoch,
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
