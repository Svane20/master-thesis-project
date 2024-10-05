import logging


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter that appends 'extra' fields like 'location', 'rotation', etc. to the log message.
    This works for any extra fields passed, making it more flexible for various log records.
    """

    def format(self, record):
        log_message = super().format(record)

        # Collect all 'extra' fields (excluding standard log fields)
        extra_fields = {key: value for key, value in record.__dict__.items()
                        if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                                       'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
                                       'message', 'msg', 'name', 'pathname', 'process', 'processName',
                                       'relativeCreated',
                                       'stack_info', 'thread', 'threadName']}

        # Format extra fields if they exist
        if extra_fields:
            extra_str = " | ".join([f"{key}: {value}" for key, value in extra_fields.items()])
            log_message = f"{log_message} | {extra_str}"

        return log_message


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level.

    Args:
        name: The name of the logger.
        level: The logging level, default is logging.INFO.

    Returns:
        The logger.
    """
    # Create a logger
    logger = logging.getLogger(name)

    # Avoid duplicate log handlers
    if not logger.hasHandlers():
        # Create a handler for outputting logs to the console
        handler = logging.StreamHandler()

        # Create a formatter and set it to the handler
        formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
