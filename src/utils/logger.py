import logging
import os
from datetime import datetime
from src.constants.constants import logs_file_dir
# Setup Log Directory
os.makedirs(logs_file_dir, exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_file_dir, LOG_FILE_NAME)

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


if __name__ == "__main__":
    logger.info("Logger test started")
    logger.warning("This is a warning log")
    logger.error("This is an error log")

    print("Log file should be created at:")
    print(LOG_FILE_PATH)
