import torch
import torch.nn as nn

from typing import Dict, Any

from training.utils.logger import setup_logging


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            optimizer_config: Dict[str, Any],
            training_config: Dict[str, Any],
    ):
        self.model = model
        self.train_config = training_config
        self.optimizer_config = optimizer_config
        self.mode = training_config["mode"]

        self.logger = setup_logging(
            __name__,
            log_level_primary="INFO",
            log_level_secondary="ERROR",
        )

        self._setup_device(training_config["accelerator"])
        self._setup_torch_backend()

        self._setup_components()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self._move_to_device(compile_model=True)

    def run(self):
        assert self.mode in ["train", "train_only", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            print("Training the model and validating.")
        elif self.mode == "val":
            print("Validating the model.")
        elif self.mode == "train_only":
            print("Training the model.")

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
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _move_to_device(self, compile_model: bool = True) -> None:
        self.logger.info(f"Moving components to device {self.device}.")

        if compile_model:
            self.model = torch.compile(self.model, backend="aot_eager")

        self.model.to(self.device)

        self.logger.info(f"Done moving components to device {self.device}.")

    def _setup_components(self):
        self.epoch = 0

        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.optimizer_config.get("amp", {}).get("enabled", False)
        )
