import logging
import sys
import os
from contextlib import contextmanager


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


class StreamToLogger:
    """
    Helper class that redirects writes from a stream (stdout/stderr) to a logger.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        # Split the buffer by newlines and log each line.
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


@contextmanager
def capture_blender_output(logger: logging.Logger, log_path: str = None, stdout_level=logging.INFO):
    """
    Temporarily redirect the OS-level stdout to the specified file.

    Args:
        logger (logging.Logger): The logger object to log messages.
        log_path (str): The path to the log file.
        stdout_level (int): The logging level for stdout messages. Defaults to logging.INFO.
    """
    target_path = log_path if log_path is not None else "blender.log"

    # Create a pipe: read_fd for reading, write_fd for output.
    read_fd, write_fd = os.pipe()

    # Save the original stdout and stderr file descriptors.
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout_fd = os.dup(stdout_fd)
    old_stderr_fd = os.dup(stderr_fd)

    try:
        # Open the target file for writing.
        with open(target_path, 'w') as target_file:
            # Redirect both stdout and stderr to the target file.
            os.dup2(target_file.fileno(), stdout_fd)
            os.dup2(target_file.fileno(), stderr_fd)
            yield  # Execute the code block with redirected output.
    finally:
        # Restore the original stdout and stderr.
        os.dup2(old_stdout_fd, stdout_fd)
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        # Read the captured output and log it to both the logger and console.
        try:
            with open(target_path, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    logger.log(stdout_level, line)
        except Exception as e:
            logger.error("Error reading captured output: %s", e)


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

    # Delete the log file if it already exists
    if os.path.exists(log_path):
        os.remove(log_path)

    # Set up the file handler if saving logs is enabled
    if save_logs:
        if not log_path:
            log_path = "run.log"
            logger.warning(f"Log path not provided. Using default log path '{log_path}'.")

        # Create the log directory
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Add the file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)

    # Set the logger as the root logger
    logging.root = logger
