from configuration.base import get_configurations
from custom_logging.custom_logger import setup_logging

if __name__ == '__main__':
    # Setup logging
    setup_logging(__name__)

    # Load configuration
    configuration = get_configurations()
