# src/utils/logging.py
"""
logging.py

Lightweight logging utility for Trial Eligibility ML.

Design goals:
- No external dependencies
- Works on locked-down Windows environments
- Sensible defaults
- Explicit configuration
- Suitable for batch or local Flask execution

This module provides a single helper:
    get_logger(name, level)
"""

import logging
from pathlib import Path


# ---------------------------------------------------------------------
# Default log format
# ---------------------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------

def get_logger(
    name,
    level="INFO",
    log_file=None,
):
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
    log_file : Optional[Path]
        Optional path to a log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
