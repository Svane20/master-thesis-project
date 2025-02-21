import logging
import sys


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter that appends 'extra' fields like 'location', 'rotation', etc., to the log message.
    This works for any extra fields passed, making it flexible for various log records and allowing richer context in logs.
    """

    # ANSI escape codes for white text
    WHITE = "\033[97m"
    RESET = "\033[0m"

    def format(self, record):
        log_message = super().format(record)

        # Collect all 'extra' fields, excluding standard log fields
        standard_fields = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'message', 'msg', 'name', 'pathname', 'process', 'processName',
            'relativeCreated', 'stack_info', 'thread', 'threadName'
        }

        # Filter out the standard log fields, leaving only the custom 'extra' fields
        extra_fields = {key: value for key, value in record.__dict__.items() if key not in standard_fields}

        # Format extra fields into a string if they exist
        if extra_fields:
            extra_str = " | ".join([f"{key}: {value}" for key, value in extra_fields.items()])
            log_message = f"{log_message} | {extra_str}"

        if record.levelno == logging.INFO:
            return f"{self.WHITE}{log_message}{self.RESET}"

        return log_message


def setup_logging(name: str, log_path: str = None, save_logs: bool = False) -> None:
    """
    Setup logging for the logger object.

    Args:
        name (str): The name of the logger.
        log_path (str): The path to the log file.
        save_logs (bool): Whether to save logs to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel("INFO")

    # Create formatter
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # Set up the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    logger.addHandler(console_handler)

    # Set up the file handler if saving logs is enabled
    if save_logs:
        if not log_path:
            log_path = "run.log"
            logger.warning(f"Log path not provided. Using default log path '{log_path}'.")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)

    # Set the logger as the root logger
    logging.root = logger
