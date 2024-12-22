import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import logging
import os.path
import numpy as np

from training.criterions import CORE_LOSS_KEY, MattingLoss
from training.early_stopping import EarlyStopping
from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.logger import setup_logging, Logger, WeightAndBiasesConfig
from training.utils.train_utils import get_amp_type, get_resume_checkpoint, DurationMeter, makedir, AverageMeter, \
    MemMeter, Phase, ProgressMeter, Meter, human_readable_time

from unet.configuration.training.base import TrainConfig


class Trainer:
    """
    Trainer class for single GPU training.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
            training_config: TrainConfig,
            meters: Optional[Dict[str, Any]] = None,
            val_epoch_freq: int = 1,
    ) -> None:
        """
        Args:
            model (nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler): Scheduler for training.
            train_data_loader (torch.utils.data.DataLoader): Data loader for training.
            test_data_loader (torch.utils.data.DataLoader): Data loader for testing.
            training_config (TrainConfig): Configuration for training.
            meters (Optional[Dict[str, Any]]): Meters for training. Default is None.
            val_epoch_freq (int): Frequency of validation. Default is 1.
        """
        # Timers
        self._setup_timers()

        # Configuration
        self.train_config = training_config
        self.optimizer_config = training_config.optimizer
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
        self._setup_components(model, optimizer, scheduler)

        # Move components to device
        self._move_to_device(compile_model=self.train_config.compile_model)

        # Data loaders
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        # Timers
        self.time_elapsed_meter = DurationMeter(name="Time Elapsed", device=self.device, fmt=":.2f")

        # Load checkpoint
        self.load_checkpoint()

    def run(self) -> None:
        """
        Run the training.
        """
        train_data_loader = self.train_data_loader
        test_data_loader = self.test_data_loader

        try:
            for epoch in range(self.epoch, self.max_epochs):
                train_metrics, train_losses = self._train_one_epoch(train_data_loader)
                test_metrics, test_losses = self._test_one_epoch(test_data_loader)
                validation_metric = test_metrics.get(
                    f"{Phase.TEST}_{self.early_stopping_config.monitor}"
                    if self.early_stopping_config.enabled
                    else CORE_LOSS_KEY,
                    None
                )

                # Step the scheduler
                self._scheduler_step(self.scheduler, validation_metric)

                # Combine train and test losses
                combined_losses = {**train_losses, **test_losses}

                # Log metrics
                epoch_duration_est = self.est_epoch_time[Phase.TRAIN] + self.est_epoch_time[Phase.TEST]
                payload = {
                    "overview/epoch": epoch,
                    "overview/epoch_duration": epoch_duration_est,
                    "overview/learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"test/{k}": v for k, v in test_metrics.items()},
                    **{f"{k}": v for k, v in combined_losses.items()}
                }
                self.logger.log_dict(
                    payload=payload,
                    step=epoch
                )
                logging.info(payload)

                # Early stopping
                current_epoch = epoch + 1
                if self.early_stopping is not None:
                    # Update early stopping
                    self.early_stopping.step(validation_metric)

                    # Check for improvement
                    if self.early_stopping.has_improved:
                        self._save_checkpoint(epoch=current_epoch)

                    # Check for early stopping
                    if self.early_stopping.should_stop:
                        logging.info(
                            f"Early stopping activated. Best score: {self.early_stopping.best_score:.4f}"
                        )
                        break
                else:
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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=1.0,
                    norm_type=2.0
                )

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

    def _test_one_epoch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Test the model for one epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for testing.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Testing metrics and losses.
        """
        # Init stat meters
        batch_time_meter = AverageMeter(name="Batch Time", device=str(self.device), fmt=":.2f")
        data_time_meter = AverageMeter(name="Data Time", device=str(self.device), fmt=":.2f")
        mem_meter = MemMeter(name="Mem (GB)", device=str(self.device), fmt=":.2f")
        data_times = []
        phase = Phase.TEST

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
            prefix="Test | Epoch: [{}]".format(self.epoch),
        )

        # Model testing loop
        self.model.eval()
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
                with torch.no_grad():
                    # Use autocast for mixed precision if enabled
                    with torch.amp.autocast(
                            device_type=self.device.type,
                            enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available(),
                            dtype=get_amp_type(self.optimizer_config.amp.amp_dtype)
                    ):
                        # Run a single step
                        loss_dict, extra_losses = self._step(X, y, phase)

                        assert len(loss_dict) == 1, f"Expected a single loss, got {len(loss_dict)} losses."
                        _, loss = loss_dict.popitem()

                        loss_meter.update(val=loss.item(), n=1)
                        for k, v in extra_losses.items():
                            if k not in extra_losses_meters:
                                extra_losses_meters[k] = AverageMeter(
                                    name=k, device=str(self.device), fmt=":.2e"
                                )
                            extra_losses_meters[k].update(val=v.item(), n=1)

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
                logging.error(f"Error during testing: {e}")
                raise e

        # Estimate epoch time
        self.est_epoch_time[phase] = batch_time_meter.avg * iters_per_epoch

        # Compute average loss metrics
        metrics[f"{phase}_loss"] = loss_meter.avg
        for k, v in extra_losses_meters.items():
            losses[k] = v.avg

        # Compute average state metrics
        metrics["data_time"] = data_time_meter.avg
        metrics["batch_time"] = batch_time_meter.avg
        metrics["mem"] = mem_meter.avg
        metrics["est_epoch_time"] = self.est_epoch_time[phase]

        logging.info(f"Test metrics: {metrics}")

        # Reset meters
        self._reset_meters([phase])

        return metrics, losses

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

        # Convert the outputs to the alpha matte probabilities
        probabilities = torch.sigmoid(outputs)

        # Calculate losses
        losses = self.criterion(probabilities, targets)

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

    def _setup_components(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler
    ) -> None:
        """
        Set up the components for training.

        Args:
            model (nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler): Scheduler for training.
        """
        logging.info("Setting up components: Model, criterion, optimizer, scheduler.")

        # Iterations
        self.epoch = 0
        self.max_epochs = self.train_config.max_epochs

        # Logger
        self.logger = self._setup_logging()

        # Components
        self.model = model
        print_model_summary(self.model, self.logging_config.log_directory)

        self.criterion = MattingLoss(
            weight_dict=self.train_config.criterion.weight_dict,
            dtype=torch.float16 if self.optimizer_config.amp.enabled else torch.float32,
            device=self.device
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Mixed precision and early stopping
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.optimizer_config.amp.enabled and torch.cuda.is_available()
        )
        self.early_stopping = self._setup_early_stopping()

        # Meters
        self.meters = self._setup_meters()
        self.best_meter_values = {}

        logging.info("Finished setting up components: Model, criterion, optimizer, scheduler")

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
        wandb_config = WeightAndBiasesConfig(
            epochs=self.train_config.max_epochs,
            learning_rate=self.optimizer_config.lr,
            learning_rate_decay=self.optimizer_config.weight_decay,
            seed=self.train_config.seed,
            device=self.train_config.accelerator,
        )

        return Logger(self.logging_config, wandb_config)

    def _move_to_device(self, compile_model: bool = True) -> None:
        """
        Move the components to the device.

        Args:
            compile_model (bool): Compile the model for faster training. Default is True.
        """
        logging.info(f"Moving components to device {self.device}.")

        if compile_model:
            logging.info("Compiling the model with backend 'aot_eager'.")

            self.model = torch.compile(self.model, backend="aot_eager")

            logging.info("Done compiling model with backend 'aot_eager'.")

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

    def _setup_timers(self) -> None:
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.TEST], 0)

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
