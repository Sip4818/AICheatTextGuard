# import logging
# import os
# from datetime import datetime

# LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# LOG_FILE_DIR = os.path.join(os.curdir, "logs")
# os.makedirs(LOG_FILE_DIR, exist_ok=True)
# LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

# # Create formatter used for both handlers
# formatter = logging.Formatter(
#     " [ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
# )

# # File handler
# file_handler = logging.FileHandler(LOG_FILE_PATH)
# file_handler.setFormatter(formatter)

# # Console handler
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)

# # Root logger setup
# logging.basicConfig(
#     level=logging.INFO,
#     handlers=[file_handler, console_handler]
# )
import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_DIR = os.path.join(os.curdir, "logs")
os.makedirs(LOG_FILE_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter(" [ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add both handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
