import logging
import os
from datetime import datetime

# Setup Log Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_FILE_DIR, exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

# Logger Setup
logger = logging.getLogger("aitextguard")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logs

# Clear any existing handlers to prevent duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# File Handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter(
    " [ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add Handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
