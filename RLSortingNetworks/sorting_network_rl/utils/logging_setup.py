import logging
import os
from typing import Optional

def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """Configures logging to both console and a file.

    Args:
        log_file (str): Path to the log file.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'), # Append mode
            logging.StreamHandler() # Console output
        ]
    )