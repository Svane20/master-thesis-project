import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
from typing import Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm
import wandb
from pydantic import BaseModel, Field
import json

from training.early_stopping import EarlyStopping
from training.metrics import calculate_dice_score, calculate_dice_edge_score, calculate_iou_score
from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.logger import setup_logging
from training.utils.train_utils import get_amp_type, get_resume_checkpoint

from unet.configuration.training import TrainConfig


class WeightAndBiasesConfig(BaseModel):
    """
    Configuration class for Weights & Biases initialization.
    """
    epochs: int = Field(description="Number of training epochs")
    learning_rate: float = Field(description="Learning rate for the optimizer")
    learning_rate_decay: float = Field(description="Learning rate decay for the optimizer")
    seed: int = Field(description="Random seed for reproducibility")
    device: str = Field(description="Device type, e.g., 'cuda' or 'cpu'")


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
        self.early_stopping_config = training_config.early_stopping
        self.checkpoint_config = training_config.checkpoint

        # Iterations
        self.epoch = 0
        self.num_epochs = self.train_config.num_epochs

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
        self._setup_early_stopping()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self._move_to_device(compile_model=self.train_config.compile_model)

        # Data loaders
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        # Load checkpoint
        self.load_checkpoint()

        # W&B initialization
        self._init_wandb()

    def run(self) -> None:
        disable_progress_bar = False
        training_start_time = time.time()

        # Data loaders
        train_data_loader = self.train_data_loader
        test_data_loader = self.test_data_loader

        try:
            for epoch in tqdm(range(self.epoch, self.num_epochs), disable=disable_progress_bar):
                start_epoch_time = time.time()

                current_epoch = epoch + 1

                # Train and test for one epoch
                train_metrics, train_times = self._train_one_epoch(
                    train_data_loader,
                    current_epoch,
                    disable_progress_bar
                )
                test_metrics, test_times = self._test_one_epoch(test_data_loader, current_epoch, disable_progress_bar)
                validation_metric = test_metrics.get(
                    self.early_stopping_config.monitor if self.early_stopping_config.enabled else "loss",
                    None
                )

                # Step the scheduler
                self._scheduler_step(self.scheduler, validation_metric)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Epoch duration
                epoch_duration = time.time() - start_epoch_time

                # Log metrics to Weights & Biases
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "epoch": current_epoch,
                        "epoch_duration": epoch_duration,
                        "learning_rate": current_lr,
                        **{f"train/{k}": v for k, v in train_times.items()},
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        **{f"test/{k}": v for k, v in test_times.items()},
                        **{f"test/{k}": v for k, v in test_metrics.items()}
                    })

                # Print metrics
                train_metrics_str = " | ".join([f"Train {k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()])
                test_metrics_str = " | ".join([f"Test {k.capitalize()}: {v:.4f}" for k, v in test_metrics.items()])
                self.logger.info(
                    f"Epoch: {current_epoch}/{self.num_epochs} | LR: {current_lr:.6f} | Epoch Duration: {epoch_duration:.2f}s\n"
                    f"{train_metrics_str} | {test_metrics_str}"
                )

                # Early stopping
                if self.early_stopping is not None:
                    # Update early stopping
                    self.early_stopping.step(validation_metric)

                    # Check for improvement
                    if self.early_stopping.has_improved:
                        self._save_checkpoint(current_epoch, validation_metric)

                    # Check for early stopping
                    if self.early_stopping.should_stop:
                        self.logger.info(
                            f"[INFO] Early stopping activated. Best score: {self.early_stopping.best_score:.4f}")
                        break
                else:
                    # Save the model at the end of each epoch
                    self._save_checkpoint(current_epoch, validation_metric)

            # Total Training Time
            total_training_time = time.time() - training_start_time
            self.wandb_run.log({"training_time": total_training_time / 60})
            self.logger.info(f"[INFO] Total training time: {total_training_time / 60:.2f} minutes")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            if self.wandb_run is not None:
                self.wandb_run.finish()
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def load_checkpoint(self) -> None:
        checkpoint_path = get_resume_checkpoint(self.checkpoint_config.resume_from)
        if checkpoint_path is not None:
            self._load_resuming_checkpoint(checkpoint_path)

    def _train_one_epoch(
            self,
            data_loader: torch.utils.data.DataLoader,
            epoch: int,
            disable_progress_bar: bool
    ) -> (Dict[str, float], Dict[str, float]):
        """
        Train the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for training.
            epoch (int): Current epoch.
            disable_progress_bar (bool): Disable the progress bar

        Returns:
            Dict[str, float]: Metrics for training.
            Dict[str, float]: Timing metrics for the epoch.
        """
        self.model.train()
        total_metrics = {}
        num_batches = 0

        # Timing
        epoch_start_time = time.time()
        data_loading_time = 0.0
        step_time = 0.0

        progress_bar = tqdm(
            enumerate(data_loader),
            desc=f"Training Epoch {epoch}/{self.num_epochs}",
            total=len(data_loader),
            disable=disable_progress_bar
        )

        for batch_idx, (X, y) in progress_bar:
            batch_start_time = time.time()
            num_batches += 1

            # Data loading time
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            data_loading_time += time.time() - batch_start_time

            # Step time
            step_start_time = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self._run_step(X, y)
            step_time += time.time() - step_start_time

            # Update metrics
            self._update_metrics(total_metrics, loss_dict)

            # Update progress bar
            current_metrics = {f"train_{k}": (total_metrics[k] / num_batches) for k in total_metrics}
            progress_bar.set_postfix(current_metrics)

        # Compute average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics["epoch"] = epoch

        # Save metrics
        self._save_metrics(avg_metrics, epoch=epoch, is_train=True)

        # Epoch duration
        epoch_duration = time.time() - epoch_start_time
        self.logger.info(
            f"Data Loading: {data_loading_time:.2f}s | Step: {step_time:.2f}s | Duration: {epoch_duration:.2f}s"
        )

        return avg_metrics, {
            "data_loading_time": data_loading_time,
            "step_time": step_time,
            "epoch_duration": epoch_duration,
        }

    def _test_one_epoch(
            self,
            data_loader: torch.utils.data.DataLoader,
            epoch: int,
            disable_progress_bar: bool
    ) -> (Dict[str, float], Dict[str, float]):
        """
        Test the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for training.
            epoch (int): Current epoch.
            disable_progress_bar (bool): Disable the progress bar

        Returns:
            Dict[str, float]: Metrics for training.
            Dict[str, float]: Timing metrics for the epoch.
        """
        self.model.eval()
        total_metrics = {}
        num_batches = 0

        # Timing
        epoch_start_time = time.time()
        data_loading_time = 0.0
        step_time = 0.0

        progress_bar = tqdm(
            enumerate(data_loader),
            desc=f"Testing Epoch {epoch}/{self.num_epochs}",
            total=len(data_loader),
            disable=disable_progress_bar
        )

        for batch_idx, (X, y) in progress_bar:
            batch_start_time = time.time()
            num_batches += 1

            # Data loading time
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            data_loading_time += time.time() - batch_start_time

            with torch.inference_mode():
                # Use autocast for mixed precision if enabled
                with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                        dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
                ):
                    # Step time
                    step_start_time = time.time()
                    loss_dict = self._step(X, y)
                    step_time += time.time() - step_start_time

                    # Update metrics
                    self._update_metrics(total_metrics, loss_dict)

            # Update the progress bar
            current_metrics = {f"test_{k}": (total_metrics[k] / num_batches) for k in total_metrics}
            progress_bar.set_postfix(current_metrics)

        # Compute average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics["epoch"] = epoch

        # Save metrics
        self._save_metrics(avg_metrics, epoch=epoch, is_train=False)

        # Epoch duration
        epoch_duration = time.time() - epoch_start_time
        self.logger.info(
            f"Data Loading: {data_loading_time:.2f}s | Step: {step_time:.2f}s | Duration: {epoch_duration:.2f}s"
        )

        return avg_metrics, {
            "data_loading_time": data_loading_time,
            "step_time": step_time,
            "epoch_duration": epoch_duration,
        }

    def _run_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """
        Run a single step of training.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target data.

        Returns:
            Dict[str, Any]: Loss dictionary.
        """
        """
        Orchestrates forward and if training, backward steps.
        """
        loss_dict = self._forward_step(X, y)
        loss = loss_dict["loss"]

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss_dict

    def _forward_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """
        Performs a forward pass and computes loss and metrics.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target data.

        Returns:
            Dict[str, Any]: Loss dictionary.
        """
        with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
        ):
            return self._step(X, y)

    def _step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate the loss and metrics for the current batch.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target data.

        Returns:
            Dict[str, Any]: Loss dictionary.
        """
        # Forward pass
        outputs = self.model(X)

        # Calculate loss
        loss = self.criterion(outputs, y)

        # Calculate predictions
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, y)

        # Loss dictionary
        loss_dict = {"loss": loss, **metrics}

        return loss_dict

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate the metrics for the predictions.

        Args:
            predictions (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Targets for the predictions.

        Returns:
            Dict[str, float]: Metrics for the predictions.
        """
        dice_score = calculate_dice_score(predictions, targets)
        dice_edge_score = calculate_dice_edge_score(predictions, targets)
        iou_score = calculate_iou_score(predictions, targets)

        return {
            "dice": dice_score,
            "dice_edge": dice_edge_score,
            "iou": iou_score
        }

    def _load_resuming_checkpoint(self, checkpoint_path: Path) -> None:
        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)

        load_state_dict_into_model(model=self.model, state_dict=checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]

        if self.optimizer_config.amp.enabled and torch.cuda.is_available() and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        if self.early_stopping is not None and "early_stopping" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        # Ensure the directory exists
        current_directory = Path(__file__).resolve().parent.parent
        save_directory = current_directory / self.checkpoint_config.save_directory
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the checkpoint
        checkpoint_path = save_directory / self.checkpoint_config.checkpoint_path

        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric
        }

        if self.scaler is not None:
            state_dict["scaler"] = self.scaler.state_dict()
        if self.early_stopping is not None:
            state_dict["early_stopping"] = self.early_stopping.state_dict()

        torch.save(obj=state_dict, f=checkpoint_path)

        self.logger.info(f"Checkpoint saved at epoch {epoch} with metric {metric:.4f}.")

    def _save_metrics(self, metrics: Dict[str, float], epoch: int, is_train: bool) -> None:
        """
        Save metrics to a JSON file.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to save.
            epoch (int): Epoch index for context.
            is_train (bool): If True, metrics are for training; otherwise, testing.
        """
        if not metrics:
            self.logger.warning("No metrics to save.")
            return

        # Set file paths
        current_directory = Path(__file__).resolve().parent
        filename = "train_metrics.json" if is_train else "test_metrics.json"
        logs_directory = current_directory / self.train_config.logging.log_directory
        logs_directory.mkdir(parents=True, exist_ok=True)

        log_entry = {"epoch": epoch, **metrics}

        try:
            with open(logs_directory / filename, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            self.logger.info(f"Metrics saved to {filename}.")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {filename}: {e}")

    def _init_wandb(self):
        if self.train_config.logging.wandb.enabled:
            wandb_config = WeightAndBiasesConfig(
                epochs=self.train_config.num_epochs,
                learning_rate=self.train_config.optimizer.lr,
                learning_rate_decay=self.train_config.optimizer.weight_decay,
                seed=self.train_config.seed,
                device=self.train_config.accelerator,
            )

            # Initialize W&B
            self.wandb_run = wandb.init(
                project=self.train_config.logging.wandb.project,
                entity=self.train_config.logging.wandb.entity,
                tags=self.train_config.logging.wandb.tags,
                notes=self.train_config.logging.wandb.notes,
                group=self.train_config.logging.wandb.group,
                job_type=self.train_config.logging.wandb.job_type,
                config=wandb_config.model_dump()
            )

            # Watch the model
            self.wandb_run.watch(
                self.model,
                self.optimizer,
                log=self.train_config.logging.log,
                log_freq=self.train_config.logging.log_freq
            )

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

    def _move_to_device(self, compile_model: bool = True) -> None:
        """
        Move the components to the device.

        Args:
            compile_model (bool): Compile the model for faster training. Default is True.
        """
        self.logger.info(f"Moving components to device {self.device}.")

        if compile_model:
            self.logger.info("Compiling the model with backend 'aot_eager'.")

            self.model = torch.compile(self.model, backend="aot_eager")

            self.logger.info("Done compiling model with backend 'aot_eager'.")

        self.model.to(self.device)

        self.logger.info(f"Done moving components to device {self.device}.")

    def _setup_torch_backend(self) -> None:
        """
        Set up the torch backend for training.
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable if input sizes are constant
            torch.backends.cudnn.deterministic = False  # Set False for better performance

            # Mixed precision optimizations
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for Ampere GPUs
            torch.backends.cudnn.allow_tf32 = True  # Enable TF32 in CuDNN
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Enable FP16 reductions
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # Enable BF16 reductions

    def _setup_scaler(self) -> None:
        """
        Set up the gradient scaler for training.
        """
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available()
        )

    def _setup_early_stopping(self) -> None:
        """
        Set up the early stopping for training.
        """
        if self.early_stopping_config.enabled:
            self.early_stopping = EarlyStopping(
                patience=self.early_stopping_config.patience,
                min_delta=self.early_stopping_config.min_delta,
                verbose=self.early_stopping_config.verbose,
                mode=self.early_stopping_config.mode
            )

    def _scheduler_step(self, scheduler: torch.optim.lr_scheduler, metric: Optional[float] = None) -> None:
        """
        Steps the scheduler based on its type and current training phase.

        Args:
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            metric (Optional[float]): Validation metric for ReduceLROnPlateau.
        """
        if isinstance(scheduler, ReduceLROnPlateau) and metric is not None:
            scheduler.step(metric)
        elif not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

    def _update_metrics(self, total_metrics: Dict[str, float], new_metrics: Dict[str, Any]) -> None:
        """
        Accumulate metrics from the current batch into total_metrics.

        Args:
            total_metrics (Dict[str, float]): Total metrics.
            new_metrics (Dict[str, Any]): New metrics.
        """
        for metric_name, metric_value in new_metrics.items():
            # Ensure metric_value is a float or a scalar
            if hasattr(metric_value, "item"):
                metric_value = metric_value.item()

            if metric_name not in total_metrics:
                total_metrics[metric_name] = metric_value
            else:
                total_metrics[metric_name] += metric_value
