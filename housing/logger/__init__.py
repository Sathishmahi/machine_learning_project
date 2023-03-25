import logging
import os
from datetime import datetime

LOGGING_DIR = "logs"
FILE_NAME = f'log_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
os.makedirs(LOGGING_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGGING_DIR, FILE_NAME)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format="[%(asctime)s]-%(name)s-%(levelname)s-%(message)s- line no [%(lineno)d] - file name [%(filename)s]",
    level=logging.INFO,
)
# print(logging.info("hello"))
