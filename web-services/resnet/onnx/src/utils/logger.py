import logging


def setup_logging(global_level=logging.INFO):
    """
    Set up logging configuration for the application.

    :param global_level: The global logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Remove existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Configure a single console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(global_level)

    # Create and set a formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    root_logger.addHandler(console_handler)

    # Set the global logging level
    root_logger.setLevel(global_level)
