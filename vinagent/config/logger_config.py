import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = "vinagent_logger",
    log_file: str = "vinagent_analysis.log",
    level: int = logging.INFO,
    log_dir: str = "logs",
    enable_console: bool = True
) -> logging.Logger:
    """
    Setup and return a logger instance with file & optional console handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear previous handlers (important in dev servers)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
