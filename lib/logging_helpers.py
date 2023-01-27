import logging
import sys
import datetime
import os


def configure_logger(to_file=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if to_file:
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(
            "logs/" + datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S") + ".log"
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
