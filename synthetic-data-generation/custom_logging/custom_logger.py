import logging


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter that appends 'extra' fields like 'location', 'rotation', etc., to the log message.
    This works for any extra fields passed, making it flexible for various log records and allowing richer context in logs.
    """

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

        return log_message


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and logging level.

    Args:
        name (str): The name of the logger.
        level (int): The logging level, default is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers to the same logger instance
    if not logger.hasHandlers():
        # Create a handler that outputs log records to the console
        handler = logging.StreamHandler()

        # Set the custom formatter for the handler
        formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
