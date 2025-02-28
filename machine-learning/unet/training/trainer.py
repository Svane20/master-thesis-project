import torch
import torch.nn as nn
import wandb
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

import time
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import logging
import os.path
import numpy as np
import json
import platform

from configuration.training.root import TrainConfig
from training.criterions import CORE_LOSS_KEY, MattingLoss
from training.early_stopping import EarlyStopping
from training.optimizers import construct_optimizer, GradientClipper
from training.schedulers import SchedulerWrapper
from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.logger import setup_logging, Logger
from metrics.utils import compute_training_metrics
from training.utils.train_utils import get_amp_type, get_resume_checkpoint, DurationMeter, makedir, AverageMeter, \
    MemMeter, Phase, ProgressMeter, Meter, human_readable_time


class Trainer:
    """
    Trainer class for single GPU training.
    """

    def __init__(
            self,
            model: nn.Module,
            train_data_loader: torch.utils.data.DataLoader,
            val_data_loader: torch.utils.data.DataLoader,
            training_config: TrainConfig,
            meters: Optional[Dict[str, Any]] = None,
            val_epoch_freq: int = 1,
    ) -> None:
        """
        Args:
            model (nn.Module): Model to train.
            train_data_loader (torch.utils.data.DataLoader): Data loader for training.
            val_data_loader (torch.utils.data.DataLoader): Data loader for validation.
            training_config (TrainConfig): Configuration for training.
            meters (Optional[Dict[str, Any]]): Meters for training. Default is None.
            val_epoch_freq (int): Frequency of validation. Default is 1.
        """
        # Timers
        self._setup_timers()

        # Configuration
        self.train_config = training_config
        self.criterion_config = training_config.criterion
        self.optimizer_config = training_config.optimizer
        self.scheduler_config = training_config.scheduler
        self.logging_config = training_config.logging
        self.early_stopping_config = training_config.early_stopping
        self.checkpoint_config = training_config.checkpoint
        self.meters_conf = meters
        self.val_epoch_freq = val_epoch_freq

        # Logging
        makedir(self.logging_config.log_directory)
        setup_logging(__name__)

        # Device
        self._setup_device(training_config.accelerator)
        self._setup_torch_backend()

        # Components
        self._setup_components(model)

        # Move components to device
        self._move_to_device(compile_model=training_config.compile_model)

        # Data loaders
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        # Timers
        self.time_elapsed_meter = DurationMeter(name="Time Elapsed", device=self.device, fmt=":.2f")

        # Load checkpoint
        self.load_checkpoint()

    def run(self) -> None:
        """
        Run the training.
        """
        train_data_loader = self.train_data_loader
        val_data_loader = self.val_data_loader

        try:
            while self.epoch < self.max_epochs:
                train_metrics, train_losses = self._train_one_epoch(train_data_loader)
                val_metrics, extra_metrics, val_losses = self._validate_one_epoch(val_data_loader)

                # Validation metric
                validation_metric_key = self.early_stopping_config.monitor \
                    if self.early_stopping_config.enabled \
                    else f"losses/{Phase.VAL}_{CORE_LOSS_KEY}"
                assert validation_metric_key in val_losses, (
                    f"Validation metric {validation_metric_key} not found in val losses. "
                    f"Available keys: {val_losses.keys()}")
                validation_metric = val_losses.get(validation_metric_key, 0.0)

                # Step the scheduler
                self.scheduler.step(validation_metric)

                # Combine train and val losses
                combined_losses = {**train_losses, **val_losses}

                # Log metrics
                epoch_duration_est = self.est_epoch_time[Phase.TRAIN] + self.est_epoch_time[Phase.VAL]
                payload = {
                    "overview/epoch": self.epoch,
                    "overview/epoch_duration": epoch_duration_est,
                    "overview/learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    **{f"metrics/{k}": v for k, v in extra_metrics.items()},
                    **{f"{k}": v for k, v in combined_losses.items()}
                }
                self.logger.log_dict(
                    payload=payload,
                    step=self.epoch
                )
                logging.info(payload)

                # Save training stats to file
                if self.logging_config.log_metrics:
                    self._save_stats(filename="training_stats.json", payload=payload)

                # Early stopping
                current_epoch = self.epoch + 1
                if self.early_stopping is not None:
                    # Update early stopping
                    self.early_stopping.step(validation_metric)

                    # Check for improvement
                    if self.early_stopping.has_improved:
                        # Save best stats to file
                        if self.logging_config.log_metrics:
                            self._save_stats(filename="best_stats.json", payload=payload)

                        # Save model checkpoint
                        self._save_checkpoint(epoch=current_epoch)

                    # Check for early stopping
                    if self.early_stopping.should_stop:
                        logging.info(
                            f"Early stopping activated. Best score: {self.early_stopping.best_score:.4f}"
                        )
                        break
                else:
                    # Save model checkpoint based on the save frequency
                    if self.checkpoint_config.save_freq > 0 and (
                            int(current_epoch) % self.checkpoint_config.save_freq) == 0:
                        # Save best stats to file
                        if self.logging_config.log_metrics:
                            self._save_stats(filename="best_stats.json", payload=payload)

                        # Save the model at the end of each epoch
                        self._save_checkpoint(epoch=current_epoch)

                # Update epoch
                self.epoch += 1

            # Total Training Time
            total_training_time = self.time_elapsed_meter.val
            self.logger.log(name="total_training_time", payload=total_training_time, step=self.max_epochs)
            logging.info(f"Total training time: {human_readable_time(total_training_time)}")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            self.logger.finish()
        finally:
            self.logger.finish()

    def _save_stats(self, filename: str, payload: Dict[str, Any]) -> None:
        """
        Save the stats to a file.

        Args:
            filename (str): Filename to save the stats to.
            payload (Dict[str, Any]): Payload to save.
        """
        try:
            with open(os.path.join(self.logging_config.log_directory, filename), "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logging.error(f"Error saving stats to file {filename}: {e}")

    def load_checkpoint(self) -> None:
        """
        Load the checkpoint for resuming training.
        """
        checkpoint_path = get_resume_checkpoint(self.checkpoint_config.resume_from)
        if checkpoint_path is not None:
            self._load_resuming_checkpoint(checkpoint_path)

    def _train_one_epoch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Train the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for training.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Training metrics and losses.
        """
        # Init stat meters
        batch_time_meter = AverageMeter(name="Batch Time", device=str(self.device), fmt=":.2f")
        data_time_meter = AverageMeter(name="Data Time", device=str(self.device), fmt=":.2f")
        mem_meter = MemMeter(name="Mem (GB)", device=str(self.device), fmt=":.2f")
        data_times = []
        phase = Phase.TRAIN

        # Init loss meters
        loss_meter = AverageMeter(name="Loss", device=str(self.device), fmt=":.2e")
        extra_losses_meters = {}

        # Progress bar
        iters_per_epoch = len(data_loader)
        progress = ProgressMeter(
            num_batches=iters_per_epoch,
            meters=[
                loss_meter,
                self.time_elapsed_meter,
                batch_time_meter,
                data_time_meter,
                mem_meter,
            ],
            real_meters=self._get_meters([phase]),
            prefix="Train | Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        metrics = {}
        losses = {}
        end = time.time()

        for batch_idx, (X, y) in enumerate(data_loader):
            # Measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)

            # Move data to device
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            try:
                # Run a single step
                self._run_step(X, y, phase, loss_meter, extra_losses_meters)

                # Clipping gradients
                if self.gradient_clipper is not None:
                    self.gradient_clipper(model=self.model)

                # Step the optimizer
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()
                self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

                # Measure memory usage
                if torch.cuda.is_available():
                    mem_meter.update(reset_peak_usage=True)

                # Update the progress bar
                if batch_idx % self.logging_config.log_freq == 0:
                    progress.display(batch_idx)
            except Exception as e:
                logging.error(f"Error during training: {e}")
                raise e

        # Estimate epoch time
        self.est_epoch_time[phase] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(phase)

        # Compute average loss metrics
        metrics[f"{phase}_loss"] = loss_meter.avg
        for k, v in extra_losses_meters.items():
            losses[k] = v.avg

        # Compute average state metrics
        metrics["est_epoch_time"] = self.est_epoch_time[phase]
        metrics["data_time"] = data_time_meter.avg
        metrics["batch_time"] = batch_time_meter.avg
        metrics["mem"] = mem_meter.avg

        logging.info(f"Train metrics: {metrics}")

        # Reset meters
        self._reset_meters([phase])

        return metrics, losses

    def _validate_one_epoch(
            self,
            data_loader: torch.utils.data.DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Validate the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for validation.

        Returns:
            Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]: Validation metrics, matting metrics and losses.
        """
        # Init stat meters
        batch_time_meter = AverageMeter(name="Batch Time", device=str(self.device), fmt=":.2f")
        data_time_meter = AverageMeter(name="Data Time", device=str(self.device), fmt=":.2f")
        mem_meter = MemMeter(name="Mem (GB)", device=str(self.device), fmt=":.2f")
        data_times = []
        phase = Phase.VAL

        # Init loss meters
        loss_meter = AverageMeter(name="Loss", device=str(self.device), fmt=":.2e")
        extra_losses_meters = {}

        # Initialize metrics accumulator
        sad_meter = AverageMeter(name="SAD", device=str(self.device), fmt=":.2f")
        mse_meter = AverageMeter(name="MSE", device=str(self.device), fmt=":.2f")
        mae_meter = AverageMeter(name="MAE", device=str(self.device), fmt=":.2f")
        grad_meter = AverageMeter(name="Grad", device=str(self.device), fmt=":.2f")

        # Progress bar
        iters_per_epoch = len(data_loader)
        progress = ProgressMeter(
            num_batches=iters_per_epoch,
            meters=[
                loss_meter,
                self.time_elapsed_meter,
                batch_time_meter,
                data_time_meter,
                mem_meter,
            ],
            real_meters=self._get_meters([phase]),
            prefix="Val | Epoch: [{}]".format(self.epoch),
        )

        # Model validation loop
        self.model.eval()
        metrics = {}
        extra_metrics = {}
        losses = {}
        end = time.time()

        for batch_idx, (X, y) in enumerate(data_loader):
            # Measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)

            # Move data to device
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            try:
                with torch.no_grad():
                    # Use autocast for mixed precision if enabled
                    with torch.amp.autocast(
                            device_type=self.device.type,
                            enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                            dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
                    ):
                        # Forward pass
                        outputs = self.model(X)

                        # Log predictions
                        if self.epoch % 10 == 0:
                            # Take 8 samples for visualization
                            sample_inputs = X[:5].detach().cpu()
                            sample_targets = X[:5].detach().cpu()
                            sample_outputs = outputs[:5].detach().cpu()

                            # Create grid images (assumes the tensors have shape [B, C, H, W])
                            produced_grid = make_grid(sample_outputs, nrow=4, normalize=True)
                            true_grid = make_grid(sample_targets, nrow=4, normalize=True)

                            # Log the images
                            self.logger.log_images_dict(
                                payload={
                                    "predictions/outputs": wandb.Image(to_pil_image(produced_grid)),
                                    "predictions/targets": wandb.Image(to_pil_image(true_grid)),
                                },
                                step=self.epoch
                            )

                        # Calculate losses
                        losses = self.criterion(outputs, y)

                        # Extract the core loss and step losses
                        loss = {}
                        step_losses = {}
                        if isinstance(losses, dict):
                            step_losses.update(
                                {f"losses/{phase}_{k}": v for k, v in losses.items()}
                            )
                            loss = losses.pop(CORE_LOSS_KEY)

                        loss_dict, extra_losses = {'loss': loss}, step_losses
                        assert len(loss_dict) == 1, f"Expected a single loss, got {len(loss_dict)} losses."
                        _, loss = loss_dict.popitem()

                        loss_meter.update(val=loss.item(), n=1)
                        for k, v in extra_losses.items():
                            if k not in extra_losses_meters:
                                extra_losses_meters[k] = AverageMeter(
                                    name=k, device=str(self.device), fmt=":.2e"
                                )
                            extra_losses_meters[k].update(val=v.item(), n=1)

                        # Calculate metrics
                        batch_metrics = compute_training_metrics(outputs, y)
                        sad_meter.update(val=batch_metrics["sad"], n=X.size(0))
                        mse_meter.update(val=batch_metrics["mse"], n=X.size(0))
                        mae_meter.update(val=batch_metrics["mae"], n=X.size(0))
                        grad_meter.update(val=batch_metrics["grad"], n=X.size(0))

                # Measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()
                self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

                # Measure memory usage
                if torch.cuda.is_available():
                    mem_meter.update(reset_peak_usage=True)

                # Update the progress bar
                if batch_idx % self.logging_config.log_freq == 0:
                    progress.display(batch_idx)
            except Exception as e:
                logging.error(f"Error during validation: {e}")
                raise e

        # Estimate epoch time
        self.est_epoch_time[phase] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(phase)

        # Compute average loss metrics
        metrics[f"{phase}_loss"] = loss_meter.avg
        for k, v in extra_losses_meters.items():
            losses[k] = v.avg

        # Compute average metrics
        extra_metrics["sad"] = sad_meter.avg
        extra_metrics["mse"] = mse_meter.avg
        extra_metrics["mae"] = mae_meter.avg
        extra_metrics["grad"] = grad_meter.avg

        # Compute average state metrics
        metrics["data_time"] = data_time_meter.avg
        metrics["batch_time"] = batch_time_meter.avg
        metrics["mem"] = mem_meter.avg
        metrics["est_epoch_time"] = self.est_epoch_time[phase]

        logging.info(f"Val metrics: {metrics}")

        # Reset meters
        self._reset_meters([phase])

        return metrics, extra_metrics, losses

    def _run_step(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            phase: str,
            loss_meter: AverageMeter,
            extra_losses_meters: Dict[str, AverageMeter]
    ) -> None:
        """
        Run a single step of training.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target data.
            phase (str): Phase of training.
            loss_meter (AverageMeter): Loss meter.
            extra_losses_meters (Dict[str, AverageMeter]): Extra loss meters.
        """
        # It's important to set grads to None, especially with Adam
        # since 0 grads will also update a model even if the step doesn't produce gradient
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
        ):
            loss_dict, extra_losses = self._step(inputs, targets, phase)

        assert len(loss_dict) == 1, f"Expected a single loss, got {len(loss_dict)} losses."
        _, loss = loss_dict.popitem()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_meter.update(val=loss.item(), n=1)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_losses_meters:
                extra_losses_meters[extra_loss_key] = AverageMeter(
                    name=extra_loss_key, device=str(self.device), fmt=":.2e"
                )
            extra_losses_meters[extra_loss_key].update(val=extra_loss.item(), n=1)

    def _step(self, inputs: torch.Tensor, targets: torch.Tensor, phase: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calculate the loss and metrics for the current batch.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target data.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Loss dictionary and step losses.
        """
        # Forward pass
        outputs = self.model(inputs)

        # Calculate losses
        losses = self.criterion(outputs, targets)

        # Extract the core loss and step losses
        loss = {}
        step_losses = {}
        if isinstance(losses, dict):
            step_losses.update(
                {f"losses/{phase}_{k}": v for k, v in losses.items()}
            )
            loss = losses.pop(CORE_LOSS_KEY)

        # Loss dictionary
        return {'loss': loss}, step_losses

    def _load_resuming_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load the checkpoint for resuming training.

        Args:
            checkpoint_path (Path): Path to the checkpoint
        """
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)

        load_state_dict_into_model(model=self.model, state_dict=checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.criterion.load_state_dict(checkpoint["criterion"])
        self.epoch = checkpoint["epoch"]
        self.ckpt_time_elapsed = checkpoint["time_elapsed"]

        if self.optimizer_config.amp.enabled and torch.cuda.is_available() and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        if self.early_stopping is not None and "early_stopping" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save the checkpoint.

        Args:
            epoch (int): Current epoch.
        """
        logging.info(f"Saving checkpoint at epoch {epoch - 1}")

        # Ensure the directory exists
        current_directory = Path(__file__).resolve().parent.parent
        save_directory = current_directory / self.checkpoint_config.save_directory
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the checkpoint
        checkpoint_path = save_directory / self.checkpoint_config.checkpoint_path

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "epoch": epoch,
            "time_elapsed": self.time_elapsed_meter.val,
        }

        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()
        if self.early_stopping is not None:
            checkpoint["early_stopping"] = self.early_stopping.state_dict()

        with open(checkpoint_path, "wb") as f:
            torch.save(obj=checkpoint, f=f)

        logging.info(f"Checkpoint saved at epoch {epoch - 1}")

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

    def _setup_components(self, model: nn.Module) -> None:
        """
        Set up the components for training.

        Args:
            model (nn.Module): Model to train.
        """
        logging.info("Setting up components: model, criterion, optimizer, scheduler, scaler.")

        # Iterations
        self.epoch = 0
        self.max_epochs = self.train_config.max_epochs

        # Logger
        self.logger = self._setup_logging()

        # Components
        self.model = model
        print_model_summary(self.model, self.logging_config.log_directory)

        # Criterion, optimizer, scheduler
        self.criterion = MattingLoss(
            weight_dict=self.criterion_config.weight_dict,
            dtype=torch.float16 if self.optimizer_config.amp.enabled else torch.float32,
            device=self.device
        )
        self.optimizer = construct_optimizer(model, self.train_config.optimizer)
        self.scheduler = SchedulerWrapper(optimizer=self.optimizer, config=self.train_config)

        # Mixed precision and early stopping
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available()
        )
        self.early_stopping = self._setup_early_stopping()

        # Gradient Clipping
        self.gradient_clipper = GradientClipper(
            enabled=self.optimizer_config.gradient_clip.enabled,
            max_norm=self.optimizer_config.gradient_clip.max_norm,
            norm_type=self.optimizer_config.gradient_clip.norm_type
        )

        # Meters
        self.meters = self._setup_meters()
        self.best_meter_values = {}

        logging.info("Finished setting up components: model, criterion, optimizer, scheduler, scaler")

    def _setup_meters(self) -> Dict[str, Any]:
        """
        Set up the meters for training.

        Returns:
            Dict[str, Any]: Meters for training.
        """
        meters = {}

        if self.meters_conf:
            for phase, phase_meters in self.meters_conf.items():
                self.meters[phase] = {}

                for key, key_meters in phase_meters.items():
                    self.meters[phase][key] = {}

                    for name, meter in key_meters.items():
                        self.meters[phase][key][name] = meter()

        return meters

    def _setup_logging(self) -> Logger:
        """
        Set up the logging for training.

        Returns:
            Logger: Logger for training.
        """
        return Logger(self.train_config)

    def _move_to_device(self, compile_model: bool = True) -> None:
        """
        Move the components to the device.

        Args:
            compile_model (bool): Compile the model for faster training. Default is True.
        """
        logging.info(f"Moving components to device {self.device}.")

        if compile_model:
            backend = "inductor" if platform.system() == "Linux" else "aot_eager"
            logging.info(f"Compiling the model with backend '{backend}'.")

            self.model = torch.compile(self.model, backend=backend)

            logging.info(f"Done compiling model with backend '{backend}'.")

        self.model.to(self.device)

        logging.info(f"Done moving components to device {self.device}.")

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

    def _setup_early_stopping(self) -> Optional[EarlyStopping]:
        """
        Set up the early stopping for training.

        Returns:
            Optional[EarlyStopping]: Early stopping for training
        """
        if self.early_stopping_config.enabled:
            return EarlyStopping(
                patience=self.early_stopping_config.patience,
                min_delta=self.early_stopping_config.min_delta,
                verbose=self.early_stopping_config.verbose,
                mode=self.early_stopping_config.mode
            )

        return None

    def _setup_timers(self) -> None:
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _log_timers(self, phase: str) -> None:
        """
        Log the timers for the given phase.

        Args:
            phase (str): Phase to log the timers for.
        """
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
                epochs_remaining * self.est_epoch_time[Phase.TRAIN]
                + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _get_meters(self, phase_filters: List[str] = None) -> Dict[str, Meter]:
        """
        Get the meters for the given phases.

        Args:
            phase_filters (List[str]): Phases to get the meters for. Default is None.

        Returns:
            Dict[str, Meter]: Meters for the given phases.
        """
        if self.meters is None:
            return {}

        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue

            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue

                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter

        return meters

    def _reset_meters(self, phases: List[str]) -> None:
        """
        Reset the meters for the given phases.

        Args:
            phases (List[str]): Phases to reset the meters for.
        """
        for meter in self._get_meters(phases).values():
            meter.reset()


def print_model_summary(model: torch.nn.Module, logging_directory: str = "") -> None:
    """
    Prints the model summary.

    Args:
        model (torch.nn.Module): Model to print the summary for.
        logging_directory (str): Directory to save the model state
    """
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if logging_directory:
        output_path = os.path.join(logging_directory, "model.txt")

        logging.info(f"Saving model summary to {output_path}")

        try:
            with open(output_path, "w") as f:
                print(model, file=f)
        except Exception as e:
            logging.error(f"Error saving model summary: {e}")

        logging.info("Model summary printed.")


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1

    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
