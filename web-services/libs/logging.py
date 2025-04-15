import logging
from pythonjsonlogger.json import JsonFormatter

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
formatter = JsonFormatter(fmt='%(asctime)s %(levelname)s %(message)s %(name)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
