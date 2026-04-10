import logging
import os
import datetime

def setup_logger(name: str, log_dir: str = "data/outputs/logs") -> logging.Logger:
    """Sets up a logger that separates phases into an appropriately named log file."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )

        # File handler (differentiable by phase/task)
        filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(os.path.join(log_dir, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
