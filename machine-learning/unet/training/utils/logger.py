import logging
import sys


def setup_logging(
        name: str,
        log_level_primary="INFO",
        log_level_secondary="ERROR",
):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level_primary.upper()))

    # Check if handlers already exist
    if not logger.hasHandlers():
        # Create formatter
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
        formatter = logging.Formatter(FORMAT)

        # Set up the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level_secondary.upper()))
        logger.addHandler(console_handler)

    return logger
