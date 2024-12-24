import logging
from typing import Dict, Any


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    """

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 0.0,
            verbose: bool = True,
            mode: str = "min"
    ) -> None:
        """
        Args:
            patience (int): Number of epochs to wait before stopping. Default is 10.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.0.
            verbose (bool): If True, prints a message for each improvement. Default is True.
            mode (str): One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored has stopped decreasing. In "max" mode, training will stop when the quantity monitored has stopped increasing. Default is "min".
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.has_improved = False

        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = float("inf")
        elif self.mode == "max":
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = float("-inf")
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'min' or 'max'.")

    def step(self, current_score: float) -> None:
        """
        Step the early stopping mechanism.

        Args:
            current_score (float): Current score to compare.
        """
        self.has_improved = False

        if self.best_score is None:
            self.best_score = current_score
            self.has_improved = True

            if self.verbose:
                logging.info(f"Initial best score set to {self.best_score}")

            return

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.has_improved = True

            if self.verbose:
                logging.info(f"Improved best score to {self.best_score}")
        else:
            self.counter += 1

            if self.verbose:
                logging.info(f"No improvement in {self.counter} epochs. Best score: {self.best_score}")

            if self.counter >= self.patience:
                self.should_stop = True

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'best_score': self.best_score,
            'counter': self.counter,
            'should_stop': self.should_stop,
            'has_improved': self.has_improved,
            'verbose': self.verbose
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): State dictionary.
        """
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.should_stop = state_dict['should_stop']
        self.has_improved = state_dict['has_improved']
        self.verbose = state_dict['verbose']

        # Reinitialize monitor_op based on mode
        if self.mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
        elif self.mode == 'max':
            self.monitor_op = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"Mode {self.mode} is not supported. Use 'min' or 'max'.")
