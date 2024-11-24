import torch
import torch.cuda.amp
import wandb
from torch import optim
from torch.optim import lr_scheduler

import argparse
import sys
import os
import traceback

from configuration.weights_and_biases import WeightAndBiasesConfig
from constants.directories import DATA_TRAIN_DIRECTORY, DATA_TEST_DIRECTORY, CHECKPOINTS_DIRECTORY
from constants.hyperparameters import BATCH_SIZE, SEED, LEARNING_RATE, NUM_EPOCHS, LEARNING_RATE_DECAY, WARMUP_EPOCHS
from constants.outputs import MODEL_OUTPUT_NAME, TRAINED_MODEL_CHECKPOINT_NAME
from dataset.data_loaders import create_data_loaders
from dataset.transforms import get_train_transforms, get_test_transforms
from model.unet import UNetV1VGG
from training import engine
from training import custom_criterions
from training.early_stopping import EarlyStopping
from utils.checkpoints import load_checkpoint
from utils.device import get_device, get_torch_compile_backend
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
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS, help="Number of epochs for training")
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


def get_combined_scheduler(
        optimizer: optim.Optimizer,
        main_scheduler: optim.lr_scheduler,
        warmup_scheduler: optim.lr_scheduler,
        warmup_epochs: int
) -> optim.lr_scheduler:
    """
    Combine main and warmup schedulers.

    Args:
        optimizer (optim.Optimizer): Optimizer for the model.
        main_scheduler (optim.lr_scheduler): Main scheduler for learning rate.
        warmup_scheduler (optim.lr_scheduler): Warmup scheduler for learning rate.
        warmup_epochs (int): Number of warmup epochs.

    Returns:
        optim.lr_scheduler: Combined scheduler.
    """

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            main_scheduler
        ],
        milestones=[warmup_epochs]
    )


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    # Setup device and torch backend
    device = get_device()
    setup_AMP()

    # Setup Weights & Biases
    configuration = WeightAndBiasesConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        learning_rate_decay=args.lr_decay,
        seed=args.seed,
        dataset="Carvana",
        architecture="U-NET",
        name_of_model=args.model_name,
        device=device.type
    )

    with wandb.init(project="U-NET", config=configuration.model_dump()) as run:
        # Get configuration
        config: WeightAndBiasesConfig = WeightAndBiasesConfig.model_validate(dict(run.config))

        # Set random seed
        set_seeds(seed=config.seed)

        # Clear the CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Instantiate Model
        model = UNetV1VGG(out_channels=1, pretrained=True).to(device)

        # Loss, Optimizer, Scheduler, AMP
        criterion = custom_criterions.BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.learning_rate_decay)
        scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

        if args.use_checkpoint:
            warmup_scheduler = None
            print("[INFO] Checkpoint loaded. Skipping warmup scheduler.")
        elif args.use_warmup:
            warmup_scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=config.warmup_epochs
            )
            print(f"[INFO] Using warmup scheduler: LinearLR with {config.warmup_epochs} warmup epochs.")
        else:
            warmup_scheduler = None
            print("[INFO] Warmup scheduler is disabled.")

        early_stopping = EarlyStopping(
            patience=10,
            min_delta=0.0,
            verbose=True,
            mode='max'
        )

        # Load Checkpoint
        start_epoch = 0
        if args.use_checkpoint:
            model, optimizer, scheduler, early_stopping, checkpoint_info = load_checkpoint(
                model=model,
                model_name=TRAINED_MODEL_CHECKPOINT_NAME,
                device=device,
                directory=CHECKPOINTS_DIRECTORY,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopping=early_stopping,
            )
            start_epoch = checkpoint_info.get("epoch", 0)

        # Compile model for faster training
        model = torch.compile(model, backend=get_torch_compile_backend())

        # Prepare data loaders
        transform = get_train_transforms()
        target_transform = get_test_transforms()
        train_data_loader, test_data_loader = create_data_loaders(
            train_directory=DATA_TRAIN_DIRECTORY,
            test_directory=DATA_TEST_DIRECTORY,
            batch_size=config.batch_size,
            transform=transform,
            target_transform=target_transform,
            num_workers=os.cpu_count() - 1,
            pin_memory=True
        )

        # Start Weights & Biases run
        run.watch(model, optimizer, log="all", log_freq=10)

        try:
            # Train the model
            engine.train(
                run=run,
                configuration=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                warmup_scheduler=warmup_scheduler,
                train_data_loader=train_data_loader,
                test_data_loader=test_data_loader,
                device=device,
                early_stopping=early_stopping,
                start_epoch=start_epoch,
            )
        except Exception as e:
            print(f"An error occurred during training: {e}")
            traceback.print_exc()
        finally:
            run.finish()


if __name__ == "__main__":
    main()
