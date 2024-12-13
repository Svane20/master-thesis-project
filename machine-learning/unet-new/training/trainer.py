import torch
import torch.nn as nn

import time
from typing import Dict, Optional

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from training.utils.logger import setup_logging
from training.utils.train_utils import get_amp_type
from unet.configuration.training import TrainConfig


class Trainer:
    """
    Trainer class for training the model.
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
            training_config: TrainConfig,
    ) -> None:
        """
        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler): Scheduler for training.
            train_data_loader (torch.utils.data.DataLoader): Data loader for training.
            test_data_loader (torch.utils.data.DataLoader): Data loader for testing.
            training_config (TrainConfig): Configuration for training.
        """
        # Configuration
        self.train_config = training_config
        self.optimizer_config = training_config.optimizer

        # Iterations
        self.epoch = 0
        self.num_epochs = self.epoch + self.train_config.num_epochs

        # Logger
        self.logger = setup_logging(
            __name__,
            log_level_primary="INFO",
            log_level_secondary="ERROR",
        )

        # Device
        self._setup_device(training_config.accelerator)
        self._setup_torch_backend()

        # Components
        self._setup_scaler()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self._move_to_device(compile_model=True)

        # Data loaders
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def run(self, run=None) -> None:
        # Clear the CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        disable_progress_bar = False
        training_start_time = time.time()

        # Data loaders
        train_data_loader = self.train_data_loader
        test_data_loader = self.test_data_loader

        for epoch in tqdm(range(self.epoch, self.num_epochs), disable=disable_progress_bar):
            start_epoch_time = time.time()

            current_epoch = epoch + 1

            train_metrics = self._train_one_epoch(train_data_loader, current_epoch, disable_progress_bar)
            test_metrics = self._test_one_epoch(test_data_loader, current_epoch, disable_progress_bar)

            # Step the scheduler
            self._scheduler_step(self.scheduler)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Calculate epoch duration
            epoch_duration = time.time() - start_epoch_time

            # Print metrics
            train_metrics_str = " | ".join([f"Train {k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()])
            test_metrics_str = " | ".join([f"Test {k.capitalize()}: {v:.4f}" for k, v in test_metrics.items()])
            self.logger.info(
                f"Epoch: {current_epoch}/{self.num_epochs} | LR: {current_lr:.6f} | Epoch Duration: {epoch_duration:.2f}s\n"
                f"{train_metrics_str} | {test_metrics_str}"
            )

        # Total Training Time
        total_training_time = time.time() - training_start_time
        self.logger.info(f"[INFO] Total training time: {total_training_time / 60:.2f} minutes")

    def _train_one_epoch(
            self,
            data_loader: torch.utils.data.DataLoader,
            epoch: int,
            disable_progress_bar: bool
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for training.
            epoch (int): Current epoch.
            disable_progress_bar (bool): Disable the progress bar

        Returns:
            Dict[str, float]: Metrics for training.
        """
        # Set the model to train mode
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            enumerate(data_loader),
            desc=f"Training Epoch {epoch}/{self.num_epochs}",
            total=len(data_loader),
            disable=disable_progress_bar
        )

        for batch_idx, (X, y) in progress_bar:
            num_batches += 1

            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                    dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
            ):
                # Forward pass
                y_pred = self.model(X)

                # Calculate loss
                loss = self.criterion(y_pred, y)

            # Zero the gradients
            self.optimizer.zero_grad()

            if self.scaler is not None:
                # Backward pass
                self.scaler.scale(loss).backward()

                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

            total_loss += loss.item()

            # Calculate predictions
            preds = torch.sigmoid(y_pred)
            preds = (preds > 0.5).float()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "train_loss": total_loss / num_batches,
                }
            )

        # Compute average metrics
        metrics = {
            "loss": total_loss / num_batches,
        }

        return metrics

    def _test_one_epoch(
            self,
            data_loader: torch.utils.data.DataLoader,
            epoch: int,
            disable_progress_bar: bool
    ) -> Dict[str, float]:
        """
        Test the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for testing.
            epoch (int): Current epoch.
            disable_progress_bar (bool): Disable the progress bar

        Returns:
            Dict[str, float]: Metrics for testing.
        """
        # Set the model to validation mode
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            enumerate(data_loader),
            desc=f"Testing Epoch {epoch}/{self.num_epochs}",
            total=len(data_loader),
            disable=disable_progress_bar
        )

        with torch.inference_mode():
            for batch_idx, (X, y) in progress_bar:
                num_batches += 1

                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                        dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
                ):
                    # Forward pass
                    y_pred = self.model(X)

                    # Calculate loss
                    loss = self.criterion(y_pred, y)

                total_loss += loss.item()

                # Calculate predictions
                preds = torch.sigmoid(y_pred)
                preds = (preds > 0.5).float()

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "test_loss": total_loss / num_batches,
                    }
                )

        # Compute average metrics
        metrics = {
            "loss": total_loss / num_batches,
        }

        return metrics

    def _setup_device(self, accelerator: str) -> None:
        """
        Set up the device for training.

        Args:
            accelerator (str): Accelerator to run the training on.
        """
        if accelerator == "cuda":
            self.device = torch.device("cuda:0")
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_torch_backend(self) -> None:
        """
        Set up the torch backend for training.
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable if input sizes are constant
            torch.backends.cudnn.deterministic = False  # Set False for better performance

            # Automatic Mixed Precision
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere GPUs

    def _move_to_device(self, compile_model: bool = True) -> None:
        """
        Move the components to the device.

        Args:
            compile_model (bool): Compile the model for faster training. Default is True.
        """
        self.logger.info(f"Moving components to device {self.device}.")

        if compile_model:
            self.model = torch.compile(self.model, backend="aot_eager")

        self.model.to(self.device)

        self.logger.info(f"Done moving components to device {self.device}.")

    def _setup_scaler(self):
        """
        Set up the gradient scaler for training.
        """
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available()
        )

    def _scheduler_step(self, scheduler: torch.optim.lr_scheduler, validation_metric: Optional[float] = None) -> None:
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
                self.logger.warn("[WARNING] Missing metric for ReduceLROnPlateau. Skipping scheduler step.")
        else:
            scheduler.step()
